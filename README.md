# FWHT

[![crates.io](https://img.shields.io/crates/v/fwht.svg)](https://crates.io/crates/fwht) [![docs.rs](https://docs.rs/fwht/badge.svg)](https://docs.rs/fwht) [![codecov](https://codecov.io/gh/edugzlez/fwht/graph/badge.svg?token=IFTI60E2FN)](https://codecov.io/gh/edugzlez/fwht)

A fast and efficient implementation of the Fast Walsh-Hadamard Transform (FWHT) in Rust.

## Features

- 🔧 **Flexible**: Works with `Vec<T>`, static arrays, and `ndarray::Array1<T>`
- 🦀 **Safe**: Leverages Rust's type system
- 📦 **No required dependencies**: ndarray support is optional
- 🧮 **Generic**: Works with any numeric type implementing `Add + Sub + Copy`

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fwht = "0.1.0"

# With ndarray support (optional)
fwht = { version = "0.1.0", features = ["ndarray"] }
```

## Quick Start

### With Vec<T>

```rust
use fwht::FWHT;

fn main() {
    // In-place modification
    let mut data = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    data.fwht_mut().unwrap();
    println!("Result: {:?}", data);

    // Create new copy
    let data = vec![1.0, 1.0, 1.0, 0.0];
    let result = data.fwht().unwrap();
    println!("Result: {:?}", result);
}
```

### With ndarray (requires "ndarray" feature)

```rust
use fwht::FWHT;
use ndarray::Array1;

fn main() {
    // In-place modification
    let mut data = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    data.fwht_mut().unwrap();
    println!("Result: {:?}", data);

    // Create new copy
    let data = Array1::from(vec![1.0, 1.0, 1.0, 0.0]);
    let result = data.fwht().unwrap();
    println!("Result: {:?}", result);
}
```

### With Static Arrays

```rust
use fwht::FWHT;

fn main() {
    // In-place modification
    let mut data = [1.0f32, 1.0, 1.0, 0.0];
    data.fwht_mut().unwrap();
    println!("Result: {:?}", data);

    // Create new copy
    let data = [1.0f64, 1.0, 1.0, 0.0];
    let result = data.fwht().unwrap();
    println!("Result: {:?}", result);
}
```

## API

### FWHT Trait

The main trait providing uniform methods for all supported types:

```rust
pub trait FWHT<T> {
    /// Apply FWHT in-place, modifying the data
    fn fwht_mut(&mut self) -> Result<(), &'static str>;

    /// Apply FWHT and return a new instance with the result
    fn fwht(&self) -> Result<Self, &'static str>;
}
```

### Implementations

- **`Vec<T>`**: Both methods available
- **`[T; N]`**: Both methods available
- **`ndarray::Array1<T>`**: Both methods available (with "ndarray" feature)

### Function-based API

For cases where the trait API isn't suitable:

- `fwht_mut<T, V>(data: &mut T)`: Works with any type implementing `AsMut<[V]>`
- `fwht<T, V>(data: &T) -> T`: Returns a new copy with the transform applied
- `fwht_slice<T>(data: &mut [T])`: Direct function for slices

### Type Requirements

Elements must implement:

- `Add<Output = T>`: For addition
- `Sub<Output = T>`: For subtraction
- `Copy`: For efficient copying

Compatible types include: `f32`, `f64`, `i32`, `i64`, complex numbers, etc.

## Advanced Examples

### Verifying the Involution Property

FWHT is its own inverse (up to scaling):

```rust
use fwht::FWHT;

let original = vec![1.0, 2.0, 3.0, 4.0];
let mut data = original.clone();

// Apply FWHT twice
data.fwht_mut().unwrap();
data.fwht_mut().unwrap();

// Scale by 1/n to recover original
let n = data.len() as f64;
for x in &mut data {
    *x /= n;
}

assert_eq!(data, original);
```

### With Different Numeric Types

```rust
use fwht::FWHT;

// With integers
let mut data_int = vec![1i32, 1, 1, 0];
data_int.fwht_mut().unwrap();

// With 32-bit floats
let data_f32 = vec![1.0f32, 1.0, 1.0, 0.0];
let result_f32 = data_f32.fwht().unwrap();
```

## Features

- `default = ["ndarray"]`: Includes ndarray support by default
- `ndarray`: Enables functions specific to `ndarray::Array1<T>`

To use without ndarray:

```toml
[dependencies]
fwht = { version = "0.1.0", default-features = false }
```

## Requirements

- **Length**: Data must have a length that is a power of 2
- **Contiguity**: For ndarray, the array must be contiguous in memory

## Tests

```bash
# All tests
cargo test

# Basic tests only (without ndarray)
cargo test --no-default-features

# Tests with ndarray
cargo test --features ndarray
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
