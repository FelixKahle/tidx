// Copyright 2026 Felix Kahle
//
// Licensed under either of
//
// Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// MIT license (LICENSE-MIT or https://opensource.org/licenses/MIT)
//
// at your option. You may not use this file except in compliance with one of these licenses.
//
// Unless required by applicable law or agreed to in writing, software distributed under these licenses is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the licenses for the specific language governing permissions and limitations under the licenses.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed dense indices: [`Idx`], a position of one kind of index, and [`IdxRange`], a run of them.
//!
//! Each kind of index is its own type, so a table keyed by one kind cannot be read with another. To the caller
//! every index is a `usize`; what it stores is the narrowest width its table needs, which its kind states.

use std::any::type_name;
use std::fmt;
use std::hash::Hash;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ops::Range;

use derive_where::derive_where;
use num_traits::{AsPrimitive, FromPrimitive, PrimInt, Unsigned, Zero};
use serde::{Deserialize, Serialize};

mod sealed {
    /// Closes [`IndexRepr`](super::IndexRepr) to the widths whose conversion to `usize` is exact.
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for usize {}
}

/// The integer an [`Idx`] is stored in: `u8`, `u16`, `u32` or `usize`.
///
/// Every value of each of these widths is a `usize` exactly, so [`Idx::index`] never loses a bit. The trait is
/// sealed, since a wider type such as `u64` would not convert exactly on every target.
pub trait IndexRepr:
    sealed::Sealed + PrimInt + Unsigned + FromPrimitive + AsPrimitive<usize> + Hash + fmt::Display
{
}

impl IndexRepr for u8 {}
impl IndexRepr for u16 {}
impl IndexRepr for u32 {}
impl IndexRepr for usize {}

/// The `usize` that `value` stands for, exactly.
#[inline]
fn to_usize<Repr>(value: Repr) -> usize
where
    Repr: IndexRepr,
{
    // `as_` is exact only where `Repr` is no wider than `usize`. This is checked at compile time for each width
    // that is used, so the seal cannot take in a wider type unnoticed.
    const { assert!(size_of::<Repr>() <= size_of::<usize>(), "an index representation should fit in `usize`") };
    value.as_()
}

/// A kind of index: its name and its storage width.
///
/// It is implemented on an uninhabited enum named for what the indices count, one for each kind. The doc of the
/// kind states the limit its width sets.
pub trait IdxKind {
    /// The name printed before the position, such as `Task` in `Task 3`.
    const NAME: &'static str;

    /// The integer an index of this kind is stored in: the narrowest of `u8`, `u16`, `u32` and `usize` that holds
    /// every position its tables can have, since a column of narrow indices is less memory to read.
    type Repr: IndexRepr;
}

/// A position in a table, of the kind of index the table is keyed by.
///
/// An `Idx<Kind>` is a 0-based index that only fits tables keyed by the same `Kind`: the index of a machine cannot
/// read the row of a task. To the caller it is a `usize`; it is stored in `Kind::Repr`. An uninhabited enum that
/// implements [`IdxKind`] names the kind and its width, and an alias names the index:
///
/// ```
/// use tidx::{Idx, IdxKind};
///
/// /// A machine of a plant, which has fewer than 65 536 machines.
/// #[derive(Debug)]
/// pub enum Machine {}
///
/// impl IdxKind for Machine {
///     const NAME: &'static str = "Machine";
///     type Repr = u16;
/// }
///
/// /// The position of a machine.
/// pub type MachineIdx = Idx<Machine>;
///
/// let machine = MachineIdx::new(3);
/// assert_eq!(machine.index(), 3);
/// assert_eq!(machine.to_string(), "Machine 3");
/// assert_eq!(MachineIdx::try_new(70_000), None);
///
/// let positions: Vec<usize> = MachineIdx::below(3).into_iter().map(MachineIdx::index).collect();
/// assert_eq!(positions, [0, 1, 2]);
/// ```
///
/// An index converts losslessly into a `usize` with `From`. A `usize` becomes an index only through
/// [`Idx::new`], which panics, [`Idx::try_new`], which refuses, or [`Idx::new_unchecked`], for a caller whose
/// invariant proves that it fits. A loop over the indices below a count walks [`Idx::below`]. Indices of one kind
/// are ordered by position. An index is serialized as its bare number.
#[derive_where(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct Idx<Kind>
where
    Kind: IdxKind,
{
    index: Kind::Repr,
    // `fn() -> Kind` keeps the index `Send`, `Sync` and covariant whatever the kind is, and owns no `Kind`.
    #[serde(skip)]
    kind: PhantomData<fn() -> Kind>,
}

impl<Kind> Idx<Kind>
where
    Kind: IdxKind,
{
    /// The index of position `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` does not fit in `Kind::Repr`.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn new(index: usize) -> Self {
        match Self::try_new(index) {
            Some(idx) => idx,
            None => panic_index_does_not_fit::<Kind>(index),
        }
    }

    /// The index of position `index`, or `None` if `index` does not fit in `Kind::Repr`.
    #[inline]
    #[must_use]
    pub fn try_new(index: usize) -> Option<Self> {
        Kind::Repr::from_usize(index).map(Self::from_repr)
    }

    /// The index of position `index`, without checking that it fits in `Kind::Repr`.
    ///
    /// # Safety
    ///
    /// `index` fits in `Kind::Repr`, the condition under which [`new`](Self::new) does not panic. The conversion
    /// assumes it without a check, so an `index` that does not fit is undefined behaviour at this call.
    #[inline]
    #[must_use]
    pub unsafe fn new_unchecked(index: usize) -> Self {
        debug_assert!(
            Kind::Repr::from_usize(index).is_some(),
            "called `Idx::new_unchecked` with index {index}, which does not fit a `{}` id (`{}`)",
            Kind::NAME,
            type_name::<Kind::Repr>(),
        );
        // SAFETY: `index` fits in `Kind::Repr`, as the caller promises, so the conversion answers `Some`.
        let index = unsafe { Kind::Repr::from_usize(index).unwrap_unchecked() };
        Self::from_repr(index)
    }

    /// Whether every index below `count` fits in `Kind::Repr`, which is whether `count - 1` does; `true` for zero.
    ///
    /// A table of `count` rows keyed by this kind of index can then name each of its rows.
    #[inline]
    #[must_use]
    pub fn fits_all_below(count: usize) -> bool {
        count == 0 || Self::try_new(count - 1).is_some()
    }

    /// Every index below `count`, `0..count`, in a range that can reach one past the largest index of the kind.
    ///
    /// # Panics
    ///
    /// Panics if the index `count - 1` does not fit in `Kind::Repr`.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn below(count: usize) -> IdxRange<Kind> {
        match Self::try_below(count) {
            Some(ids) => ids,
            None => panic_count_does_not_fit::<Kind>(count),
        }
    }

    /// Every index below `count`, `0..count`, or `None` if the index `count - 1` does not fit in `Kind::Repr`.
    #[inline]
    #[must_use]
    pub fn try_below(count: usize) -> Option<IdxRange<Kind>> {
        IdxRange::try_new(Self::first(), count)
    }

    /// Every index below `count`, `0..count`, without checking that they fit in `Kind::Repr`.
    ///
    /// A table that checked its length when it was made hands out its indices through this.
    ///
    /// # Safety
    ///
    /// The index `count - 1` fits in `Kind::Repr`, the condition under which [`below`](Self::below) does not
    /// panic: the range makes its indices without a check.
    #[inline]
    #[must_use]
    pub unsafe fn below_unchecked(count: usize) -> IdxRange<Kind> {
        debug_assert!(
            Self::fits_all_below(count),
            "called `Idx::below_unchecked` with count {count}, whose last id does not fit a `{}` id (`{}`)",
            Kind::NAME,
            type_name::<Kind::Repr>(),
        );
        // SAFETY: every index below `count` fits in `Kind::Repr`, as the caller promises.
        unsafe { IdxRange::new_unchecked(Self::first(), count) }
    }

    /// The position this index stands for.
    #[inline]
    #[must_use]
    pub fn index(self) -> usize {
        to_usize(self.index)
    }

    /// The index of position 0, which fits every width.
    fn first() -> Self {
        Self::from_repr(Kind::Repr::zero())
    }

    const fn from_repr(index: Kind::Repr) -> Self {
        Self { index, kind: PhantomData }
    }
}

/// Panics because [`Idx::new`] was handed `index`, which does not fit an index of `Kind`.
#[cold] // Out of range is the exception: the caller's code is laid out for the index that fits.
#[inline(never)] // Keeps the formatting of the message out of every caller.
#[track_caller]
fn panic_index_does_not_fit<Kind>(index: usize) -> !
where
    Kind: IdxKind,
{
    panic!(
        "called `Idx::new` with index {index}, which does not fit a `{}` id (`{}`)",
        Kind::NAME,
        type_name::<Kind::Repr>()
    )
}

/// Panics because [`Idx::below`] was handed `count`, whose last index does not fit an index of `Kind`.
#[cold] // Out of range is the exception: the caller's code is laid out for the count that fits.
#[inline(never)] // Keeps the formatting of the message out of every caller.
#[track_caller]
fn panic_count_does_not_fit<Kind>(count: usize) -> !
where
    Kind: IdxKind,
{
    panic!(
        "called `Idx::below` with count {count}, whose last id does not fit a `{}` id (`{}`)",
        Kind::NAME,
        type_name::<Kind::Repr>()
    )
}

impl<Kind> From<Idx<Kind>> for usize
where
    Kind: IdxKind,
{
    #[inline]
    fn from(idx: Idx<Kind>) -> Self {
        idx.index()
    }
}

impl<Kind> fmt::Display for Idx<Kind>
where
    Kind: IdxKind,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", Kind::NAME, self.index)
    }
}

// Written by hand: the derived form would show the phantom kind instead of its name.
impl<Kind> fmt::Debug for Idx<Kind>
where
    Kind: IdxKind,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// The indices of one kind from position `start` up to but excluding position `end`.
///
/// A range is a value: it is `Copy`, and walking it with [`IdxRange::iter`] or `for` leaves it as it was, so a
/// range kept in a table can be read any number of times. It is empty when `end <= start`. Its start is an index
/// and its end a position, so a range can end one past the largest index of its kind: [`Idx::below`] holds every
/// index of a kind, all 256 of a kind stored in `u8` included.
#[derive_where(Clone, Copy, PartialEq, Eq, Hash)]
pub struct IdxRange<Kind>
where
    Kind: IdxKind,
{
    // Invariant: every position below `end` fits in `Kind::Repr`, so every position the range holds is an index of
    // its kind. `try_new` checks it, `new_unchecked` is promised it and `From<Range>` takes an `end` that is an
    // index itself. The iterator makes its indices without a check on this.
    start: usize,
    end: usize,
    #[derive_where(skip)]
    kind: PhantomData<fn() -> Kind>,
}

impl<Kind> IdxRange<Kind>
where
    Kind: IdxKind,
{
    /// The indices from `start` up to but excluding position `end`; empty when `end <= start`.
    ///
    /// # Panics
    ///
    /// Panics if the index `end - 1` does not fit in `Kind::Repr`.
    #[inline]
    #[must_use]
    #[track_caller]
    pub fn new(start: Idx<Kind>, end: usize) -> Self {
        match Self::try_new(start, end) {
            Some(range) => range,
            None => panic_end_does_not_fit::<Kind>(end),
        }
    }

    /// The indices from `start` up to but excluding position `end`, or `None` if the index `end - 1` does not fit
    /// in `Kind::Repr`.
    #[inline]
    #[must_use]
    pub fn try_new(start: Idx<Kind>, end: usize) -> Option<Self> {
        if Idx::<Kind>::fits_all_below(end) {
            debug_assert!(Idx::<Kind>::fits_all_below(end), "the end should have been checked against the width");
            // SAFETY: every index below `end` fits in `Kind::Repr`, checked just above.
            Some(unsafe { Self::new_unchecked(start, end) })
        } else {
            None
        }
    }

    /// The indices from `start` up to but excluding position `end`, without checking that they fit in
    /// `Kind::Repr`.
    ///
    /// # Safety
    ///
    /// The index `end - 1` fits in `Kind::Repr`, the condition under which [`new`](Self::new) does not panic: the
    /// range makes its indices without a check.
    #[inline]
    #[must_use]
    pub unsafe fn new_unchecked(start: Idx<Kind>, end: usize) -> Self {
        debug_assert!(
            Idx::<Kind>::fits_all_below(end),
            "called `IdxRange::new_unchecked` with end {end}, whose last id does not fit a `{}` id (`{}`)",
            Kind::NAME,
            type_name::<Kind::Repr>(),
        );
        Self { start: start.index(), end, kind: PhantomData }
    }

    /// The position of the first index of the range.
    #[inline]
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// The position one past the last index of the range, which may lie one past the largest index of the kind.
    #[inline]
    #[must_use]
    pub const fn end(&self) -> usize {
        self.end
    }

    /// The number of indices in the range, zero when `end <= start`.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the range holds no index, which is when `end <= start`.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Whether `idx` lies in the range: `start <= idx < end`.
    #[inline]
    #[must_use]
    pub fn contains(&self, idx: Idx<Kind>) -> bool {
        (self.start..self.end).contains(&idx.index())
    }

    /// The indices of the range in ascending order, walked from either end.
    #[inline]
    pub fn iter(&self) -> IdxRangeIter<Kind> {
        IdxRangeIter::new(*self)
    }
}

/// Panics because [`IdxRange::new`] was handed `end`, whose last index does not fit an index of `Kind`.
#[cold] // Out of range is the exception: the caller's code is laid out for the end that fits.
#[inline(never)] // Keeps the formatting of the message out of every caller.
#[track_caller]
fn panic_end_does_not_fit<Kind>(end: usize) -> !
where
    Kind: IdxKind,
{
    panic!(
        "called `IdxRange::new` with end {end}, whose last id does not fit a `{}` id (`{}`)",
        Kind::NAME,
        type_name::<Kind::Repr>()
    )
}

impl<Kind> Default for IdxRange<Kind>
where
    Kind: IdxKind,
{
    /// The empty range `0..0`.
    #[inline]
    fn default() -> Self {
        Self { start: 0, end: 0, kind: PhantomData }
    }
}

// Written by hand: the derived form would show the phantom kind instead of its name.
impl<Kind> fmt::Debug for IdxRange<Kind>
where
    Kind: IdxKind,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IdxRange<{}> {}..{}", Kind::NAME, self.start, self.end)
    }
}

impl<Kind> From<Range<Idx<Kind>>> for IdxRange<Kind>
where
    Kind: IdxKind,
{
    #[inline]
    fn from(range: Range<Idx<Kind>>) -> Self {
        // `end` is an index, so every position below it fits.
        Self { start: range.start.index(), end: range.end.index(), kind: PhantomData }
    }
}

impl<Kind> IntoIterator for IdxRange<Kind>
where
    Kind: IdxKind,
{
    type Item = Idx<Kind>;
    type IntoIter = IdxRangeIter<Kind>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<Kind> IntoIterator for &IdxRange<Kind>
where
    Kind: IdxKind,
{
    type Item = Idx<Kind>;
    type IntoIter = IdxRangeIter<Kind>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// The indices of an [`IdxRange`] not yet yielded, in ascending order and from either end.
#[derive_where(Clone, Debug)]
pub struct IdxRangeIter<Kind>
where
    Kind: IdxKind,
{
    // Invariant: `start <= end`, and every position below `end` fits in `Kind::Repr`, as in the range it was made
    // from, whose `end` it keeps. `new` establishes the order and each step moves one end towards the other only
    // while they differ, so `end - start` never underflows. The unchecked making of each index rests on the fit.
    start: usize,
    end: usize,
    #[derive_where(skip)]
    kind: PhantomData<fn() -> Kind>,
}

impl<Kind> IdxRangeIter<Kind>
where
    Kind: IdxKind,
{
    /// The indices of `range`, none if `end <= start`.
    fn new(range: IdxRange<Kind>) -> Self {
        Self { start: range.start.min(range.end), end: range.end, kind: PhantomData }
    }
}

impl<Kind> Iterator for IdxRangeIter<Kind>
where
    Kind: IdxKind,
{
    type Item = Idx<Kind>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let position = self.start;
            debug_assert!(
                position < self.end && Idx::<Kind>::fits_all_below(self.end),
                "every position of the range should fit",
            );
            // SAFETY: `position` is below `end`, every position below which fits in `Kind::Repr` by the invariant
            // of `IdxRangeIter`.
            let idx = unsafe { Idx::new_unchecked(position) };
            self.start += 1;
            Some(idx)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.start;
        (len, Some(len))
    }
}

impl<Kind> DoubleEndedIterator for IdxRangeIter<Kind>
where
    Kind: IdxKind,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let position = self.end - 1;
            debug_assert!(
                position < self.end && Idx::<Kind>::fits_all_below(self.end),
                "every position of the range should fit",
            );
            // SAFETY: `position` is below `end`, every position below which fits in `Kind::Repr` by the invariant
            // of `IdxRangeIter`.
            let idx = unsafe { Idx::new_unchecked(position) };
            self.end = position;
            Some(idx)
        } else {
            None
        }
    }
}

impl<Kind> ExactSizeIterator for IdxRangeIter<Kind> where Kind: IdxKind {}

impl<Kind> FusedIterator for IdxRangeIter<Kind> where Kind: IdxKind {}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// Stored in `u8`, so the edge of its width is in reach of a test.
    #[derive(Debug)]
    enum Row {}

    impl IdxKind for Row {
        const NAME: &'static str = "Row";
        type Repr = u8;
    }

    #[derive(Debug)]
    enum Lane {}

    impl IdxKind for Lane {
        const NAME: &'static str = "Lane";
        type Repr = u16;
    }

    #[derive(Debug)]
    enum Col {}

    impl IdxKind for Col {
        const NAME: &'static str = "Col";
        type Repr = u32;
    }

    #[derive(Debug)]
    enum WideCol {}

    impl IdxKind for WideCol {
        const NAME: &'static str = "WideCol";
        type Repr = usize;
    }

    type RowIdx = Idx<Row>;
    type LaneIdx = Idx<Lane>;
    type ColIdx = Idx<Col>;
    type WideColIdx = Idx<WideCol>;

    fn positions<Rows>(iter: Rows) -> Vec<usize>
    where
        Rows: Iterator<Item = RowIdx>,
    {
        iter.map(RowIdx::index).collect()
    }

    fn rows(start: usize, end: usize) -> IdxRange<Row> {
        IdxRange::new(RowIdx::new(start), end)
    }

    /// The orders in which a walk takes its steps, each repeated until the walk is over: `true` is a step from the
    /// front and `false` one from the back.
    const STEP_PATTERNS: [&[bool]; 5] =
        [&[true], &[false], &[true, false], &[false, false, true], &[true, true, false, true]];

    /// Checks that a walk over the rows `start..end` in the order of `pattern` yields what a `usize` range yields,
    /// with the same length left after each step, and stays done for two steps past its end.
    fn assert_walk_matches_a_range_of_usize(start: usize, end: usize, pattern: &[bool]) {
        let mut iter = rows(start, end).iter();
        let mut model = start..end;
        let steps = model.len() + 2;

        for &from_front in pattern.iter().cycle().take(steps) {
            let (step, expected) =
                if from_front { (iter.next(), model.next()) } else { (iter.next_back(), model.next_back()) };
            assert_eq!(step.map(RowIdx::index), expected);
            assert_eq!(iter.len(), model.len());
        }
    }

    #[test]
    fn test_new_keeps_every_position_its_width_holds() {
        let largest_col = usize::try_from(u32::MAX).expect("a `u32` should fit in `usize`");

        assert_eq!(RowIdx::new(0).index(), 0);
        assert_eq!(RowIdx::new(255).index(), 255);
        assert_eq!(ColIdx::new(largest_col).index(), largest_col);
        assert_eq!(WideColIdx::new(usize::MAX).index(), usize::MAX);
    }

    #[test]
    fn test_try_new_refuses_a_position_past_its_width() {
        assert_eq!(RowIdx::try_new(255), Some(RowIdx::new(255)));
        assert_eq!(RowIdx::try_new(256), None);
        if let Ok(past_col) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert_eq!(ColIdx::try_new(past_col), None);
        }
    }

    #[test]
    #[should_panic(expected = "called `Idx::new` with index 256, which does not fit a `Row` id (`u8`)")]
    fn test_new_panics_past_its_width() {
        let _ = RowIdx::new(256);
    }

    #[test]
    fn test_fits_all_below_holds_up_to_one_past_the_width() {
        assert!(RowIdx::fits_all_below(0));
        assert!(RowIdx::fits_all_below(1));
        assert!(RowIdx::fits_all_below(256));
        assert!(!RowIdx::fits_all_below(257));
        assert!(!RowIdx::fits_all_below(usize::MAX));
        assert!(WideColIdx::fits_all_below(usize::MAX));
    }

    #[test]
    fn test_conversion_into_usize_keeps_the_position() {
        assert_eq!(usize::from(RowIdx::new(7)), 7);
        assert_eq!(usize::from(ColIdx::new(70_000)), 70_000);
    }

    #[test]
    fn test_storage_is_its_width() {
        assert_eq!(size_of::<RowIdx>(), 1);
        assert_eq!(size_of::<LaneIdx>(), 2);
        assert_eq!(size_of::<ColIdx>(), 4);
        assert_eq!(size_of::<WideColIdx>(), size_of::<usize>());
    }

    #[test]
    fn test_order_and_equality_follow_the_position() {
        assert!(RowIdx::new(2) < RowIdx::new(3));
        assert_eq!(RowIdx::new(3), RowIdx::new(3));
        assert_ne!(RowIdx::new(3), RowIdx::new(4));
        assert_eq!(RowIdx::new(9).max(RowIdx::new(4)), RowIdx::new(9));
    }

    #[test]
    fn test_hash_tells_positions_apart() {
        let distinct: HashSet<RowIdx> = (0..10).chain(0..10).map(RowIdx::new).collect();

        assert_eq!(distinct.len(), 10);
    }

    #[test]
    fn test_display_and_debug_name_the_kind() {
        assert_eq!(RowIdx::new(4).to_string(), "Row 4");
        assert_eq!(format!("{:?}", ColIdx::new(4)), "Col 4");
        assert_eq!(format!("{:?}", [RowIdx::new(3), RowIdx::new(1)]), "[Row 3, Row 1]");
    }

    #[test]
    fn test_serde_writes_the_bare_number() {
        let json = serde_json::to_string(&RowIdx::new(7)).expect("a row should serialize");
        assert_eq!(json, "7");
        assert_eq!(serde_json::from_str::<RowIdx>(&json).expect("a row should deserialize"), RowIdx::new(7));

        let col = ColIdx::new(70_000);
        let json = serde_json::to_string(&col).expect("a column should serialize");
        assert_eq!(serde_json::from_str::<ColIdx>(&json).expect("a column should deserialize"), col);
    }

    #[test]
    fn test_serde_refuses_a_number_past_its_width() {
        assert!(serde_json::from_str::<RowIdx>("256").is_err());
        assert!(serde_json::from_str::<RowIdx>("-1").is_err());
    }

    #[test]
    fn test_below_holds_the_first_count_indices() {
        let range = RowIdx::below(3);

        assert_eq!(range, rows(0, 3));
        assert_eq!(range.len(), 3);
        assert!(!range.is_empty());
        assert!(RowIdx::below(0).is_empty());
        assert_eq!(RowIdx::below(255).len(), 255);
    }

    #[test]
    fn test_below_reaches_every_index_of_the_width() {
        let every = RowIdx::below(256);

        assert_eq!(every.len(), 256);
        assert_eq!((every.start(), every.end()), (0, 256));
        assert!(every.contains(RowIdx::new(255)));
        assert_eq!(positions(every.iter()), (0..256).collect::<Vec<_>>());
        assert_eq!(every.iter().next_back(), Some(RowIdx::new(255)));
        assert_eq!(LaneIdx::below(65_536).len(), 65_536);
    }

    #[test]
    fn test_try_below_refuses_a_count_past_the_width() {
        assert_eq!(RowIdx::try_below(256), Some(RowIdx::below(256)));
        assert_eq!(RowIdx::try_below(257), None);
        assert_eq!(RowIdx::try_below(usize::MAX), None);
    }

    #[test]
    #[should_panic(expected = "called `Idx::below` with count 257, whose last id does not fit a `Row` id (`u8`)")]
    fn test_below_panics_past_its_width() {
        let _ = RowIdx::below(257);
    }

    #[test]
    fn test_new_reaches_past_the_largest_id_from_any_start() {
        let tail = IdxRange::new(RowIdx::new(250), 256);

        assert_eq!(tail.len(), 6);
        assert_eq!(positions(tail.iter()), [250, 251, 252, 253, 254, 255]);
        assert_eq!(IdxRange::try_new(RowIdx::new(250), 256), Some(tail));
        assert_eq!(IdxRange::try_new(RowIdx::new(250), 257), None);
    }

    #[test]
    #[should_panic(expected = "called `IdxRange::new` with end 257, whose last id does not fit a `Row` id (`u8`)")]
    fn test_range_new_panics_past_its_width() {
        let _ = IdxRange::new(RowIdx::new(0), 257);
    }

    #[test]
    fn test_contains_includes_start_and_excludes_end() {
        let range = rows(2, 5);

        assert!(!range.contains(RowIdx::new(1)));
        assert!(range.contains(RowIdx::new(2)));
        assert!(range.contains(RowIdx::new(4)));
        assert!(!range.contains(RowIdx::new(5)));
    }

    #[test]
    fn test_range_with_end_before_start_is_empty() {
        let range = rows(5, 2);

        assert!(range.is_empty());
        assert_eq!(range.len(), 0);
        assert!(!range.contains(RowIdx::new(3)));
        assert_eq!(range.iter().next(), None);
        assert_eq!(range.iter().next_back(), None);
    }

    #[test]
    fn test_default_is_the_empty_range_at_zero() {
        assert_eq!(IdxRange::<Row>::default(), rows(0, 0));
    }

    #[test]
    fn test_from_std_range_keeps_both_ends() {
        assert_eq!(IdxRange::from(RowIdx::new(1)..RowIdx::new(3)), rows(1, 3));
    }

    #[test]
    fn test_new_keeps_both_ends_as_given() {
        let range = rows(5, 2);

        assert_eq!(range.start(), 5);
        assert_eq!(range.end(), 2);
    }

    #[test]
    fn test_walking_a_range_leaves_it_as_it_was() {
        let range = rows(1, 4);

        assert_eq!(positions(range.into_iter()), [1, 2, 3]);
        assert_eq!(positions((&range).into_iter()), [1, 2, 3]);
        assert_eq!(range, rows(1, 4));
    }

    #[test]
    fn test_iter_walks_from_both_ends() {
        let mut iter = rows(0, 4).iter();

        assert_eq!(iter.next(), Some(RowIdx::new(0)));
        assert_eq!(iter.next_back(), Some(RowIdx::new(3)));
        assert_eq!(iter.len(), 2);
        assert_eq!(positions(iter.clone()), [1, 2]);
        assert_eq!(positions(iter.rev()), [2, 1]);
    }

    #[test]
    fn test_iter_reaches_the_edge_of_its_width_and_stays_done() {
        let mut iter = RowIdx::below(256).iter();
        iter.nth(249);

        assert_eq!(positions(iter.by_ref()), [250, 251, 252, 253, 254, 255]);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn test_debug_shows_the_positions() {
        let mut iter = rows(1, 3).iter();
        iter.next();

        assert_eq!(format!("{:?}", rows(1, 3)), "IdxRange<Row> 1..3");
        assert_eq!(format!("{iter:?}"), "IdxRangeIter { start: 2, end: 3, .. }");
    }

    /// `new_unchecked` gives what `new` and `try_new` give for every position that fits, and `try_new` refuses
    /// exactly the positions past the width.
    #[test]
    fn test_new_unchecked_matches_new_where_it_fits() {
        for index in 0..1024 {
            let checked = RowIdx::try_new(index);
            assert_eq!(checked.is_some(), index <= 255);

            if let Some(checked) = checked {
                debug_assert!(index <= 255, "the position should have been checked against the width");
                // SAFETY: `index` fits in `u8`, checked just above.
                let unchecked = unsafe { RowIdx::new_unchecked(index) };
                assert_eq!(unchecked, checked);
                assert_eq!(unchecked, RowIdx::new(index));
            }
        }
    }

    /// `below_unchecked` gives what `below` gives for every count whose indices fit, and `try_below` refuses
    /// exactly the counts past the width.
    #[test]
    fn test_below_unchecked_matches_below_where_it_fits() {
        for count in 0..1024 {
            let checked = RowIdx::try_below(count);
            assert_eq!(checked.is_some(), count <= 256);

            if let Some(checked) = checked {
                debug_assert!(count <= 256, "the count should have been checked against the width");
                // SAFETY: every position below `count` fits in `u8`, checked just above.
                let unchecked = unsafe { RowIdx::below_unchecked(count) };
                assert_eq!(unchecked, checked);
                assert_eq!(unchecked, RowIdx::below(count));
                assert_eq!(positions(unchecked.iter()), (0..count).collect::<Vec<_>>());
            }
        }
    }

    /// `IdxRange::new_unchecked` gives what `new` gives for every end whose indices fit, and `try_new` refuses
    /// exactly the ends past the width, from every start.
    #[test]
    fn test_range_new_unchecked_matches_new_where_it_fits() {
        for start in RowIdx::below(256) {
            for end in 0..1024 {
                let checked = IdxRange::try_new(start, end);
                assert_eq!(checked.is_some(), end <= 256);

                if let Some(checked) = checked {
                    debug_assert!(end <= 256, "the end should have been checked against the width");
                    // SAFETY: every position below `end` fits in `u8`, checked just above.
                    let unchecked = unsafe { IdxRange::new_unchecked(start, end) };
                    assert_eq!(unchecked, checked);
                    assert_eq!(unchecked, IdxRange::new(start, end));
                }
            }
        }
    }

    /// Every range of the width, walked in each of the step patterns, yields what a `usize` range yields, up to
    /// the range over every index of the width.
    #[test]
    fn test_iter_matches_a_range_of_usize() {
        for start in 0..=255 {
            for end in 0..=256 {
                for pattern in STEP_PATTERNS {
                    assert_walk_matches_a_range_of_usize(start, end, pattern);
                }
            }
        }
    }
}
