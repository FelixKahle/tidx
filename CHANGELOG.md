# Changelog

All notable changes to `tidx` are listed here. The versions follow [semantic versioning](https://semver.org).

## Unreleased

## 0.1.0 — 2026-11-25

- `Idx<Kind>`, a typed dense index stored in the width its `IdxKind` states and read as a `usize`.
- `IdxRange<Kind>`, a run of indices of one kind, and `IdxRangeIter`, which walks it from either end.
- `IndexRepr`, sealed to `u8`, `u16`, `u32` and `usize`.
- `serde` support: an index is serialized as its bare number.
