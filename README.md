# tidx

[![Crates.io](https://img.shields.io/crates/v/tidx.svg)](https://crates.io/crates/tidx)
[![Documentation](https://docs.rs/tidx/badge.svg)](https://docs.rs/tidx)
[![CI](https://github.com/FelixKahle/tidx/actions/workflows/ci.yml/badge.svg)](https://github.com/FelixKahle/tidx/actions/workflows/ci.yml)
[![Licence](https://img.shields.io/crates/l/tidx.svg)](#licence)

Typed dense indices for Rust.

Each kind of index is a type of its own, so a table keyed by one kind cannot be read with another. To the caller
every index is a `usize`; what it stores is the narrowest of `u8`, `u16`, `u32` and `usize` its table needs.

## Usage

```sh
cargo add tidx
```

```rust
use tidx::{Idx, IdxKind};

/// A machine of a plant, which has fewer than 65 536 machines.
#[derive(Debug)]
pub enum Machine {}

impl IdxKind for Machine {
    const NAME: &'static str = "Machine";
    type Repr = u16;
}

/// The position of a machine.
pub type MachineIdx = Idx<Machine>;

let machine = MachineIdx::new(3);
assert_eq!(machine.index(), 3);
assert_eq!(machine.to_string(), "Machine 3");
assert_eq!(MachineIdx::try_new(70_000), None);

let positions: Vec<usize> = MachineIdx::below(3).into_iter().map(MachineIdx::index).collect();
assert_eq!(positions, [0, 1, 2]);
```

An index is serialized with `serde` as its bare number. The crate needs Rust 1.100 or newer.

## Contributing

Changes reach `main` through a pull request. A change is done when these pass, which CI runs on every pull
request:

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
RUSTFLAGS="-D warnings" cargo test --locked
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

## Licence

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT licence](LICENSE-MIT), at your
option.
