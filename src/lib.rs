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

//! Typed dense indices.
//!
//! An [`Idx`] is a 0-based position in a table, of the kind of index the table is keyed by, so the index of one
//! table cannot read another. A kind is an uninhabited enum that implements [`IdxKind`]: it names the index and
//! states the integer it is stored in, the narrowest of `u8`, `u16`, `u32` and `usize` its table needs. To the
//! caller every index is a `usize`. An [`IdxRange`] is a run of indices of one kind, walked by an
//! [`IdxRangeIter`].
//!
//! [`Idx`] shows how a kind and its index are declared and used.

mod idx;

pub use idx::{Idx, IdxKind, IdxRange, IdxRangeIter, IndexRepr};
