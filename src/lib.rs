//! Fast Walsh-Hadamard Transform (FWHT) library
//!
//! This library provides efficient implementations of the Fast Walsh-Hadamard Transform
//! for various container types including `Vec<T>`, static arrays `[T; N]`, and
//! optionally `ndarray::Array1<T>`.
//!
//! # Features
//!
//! - **Fast**: O(n log n) implementation using butterfly operations
//! - **Generic**: Works with any numeric type implementing `Add + Sub + Copy`
//! - **Flexible**: Uniform API across different container types via traits
//! - **Optional dependencies**: ndarray support is feature-gated
//!
//! # Quick Start
//!
//! ```
//! use fwht::FWHT;
//!
//! let mut data = vec![1.0, 1.0, 1.0, 0.0];
//! data.fwht_mut().unwrap();
//! assert_eq!(data, vec![3.0, 1.0, 1.0, -1.0]);
//!
//! let data2 = vec![1.0, 1.0, 1.0, 0.0];
//! let result = data2.fwht().unwrap();
//! assert_eq!(result, vec![3.0, 1.0, 1.0, -1.0]);
//! assert_eq!(data2, vec![1.0, 1.0, 1.0, 0.0]); // Original unchanged
//!
//! # #[cfg(feature = "ndarray")]
//! # {
//! // With ndarray (requires "ndarray" feature)
//! use ndarray::Array1;
//! let data = Array1::from(vec![1.0, 1.0, 1.0, 0.0]);
//! let result = data.fwht();
//! # }
//! ```
//!
//! # API Overview
//!
//! ## Trait-based API (Recommended)
//!
//! The main interface is the [`FWHT`] trait which provides:
//!
//! - [`FWHT::fwht_mut`]: In-place transformation (memory efficient)
//! - [`FWHT::fwht`]: Returns a new container with the result
//!
//! ## Function-based API
//!
//! The original functions are also available:
//!
//! - [`fwht_mut`]: In-place transformation for any `AsMut<[T]>`
//! - [`fwht`]: Copy-based transformation for any `Clone + AsMut<[T]>`
//!
//! # Requirements
//!
//! - Container length must be a power of 2
//! - Elements must implement `Add<Output = T> + Sub<Output = T> + Copy`
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use fwht::FWHT;
//!
//! let data = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
//! let result = data.fwht().unwrap();
//! assert_eq!(result, vec![4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]);
//! ```
//!
//! ## Involution Property
//!
//! FWHT is its own inverse (up to scaling):
//!
//! ```
//! use fwht::FWHT;
//!
//! let original = vec![1.0, 2.0, 3.0, 4.0];
//! let mut data = original.clone();
//!
//! // Apply FWHT twice
//! data.fwht_mut().unwrap();
//! data.fwht_mut().unwrap();
//!
//! // Scale by 1/n to recover original
//! let n = data.len() as f64;
//! for x in &mut data { *x /= n; }
//!
//! // Should equal original (within floating point precision)
//! for (a, b) in data.iter().zip(original.iter()) {
//!     assert!((a - b).abs() < 1e-10);
//! }
//! ```

// Core algorithm
pub mod core;

// Trait definitions
pub mod traits;

// Implementations for different types
pub mod impls;

pub mod functions;

pub use functions::{fwht, fwht_mut};
pub use traits::FWHT;

pub use core::{fwht_slice, is_valid_fwht_length, next_power_of_two};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_trait_and_function_consistency() {
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let result_trait = data.fwht();
        let result_function = fwht(&data);

        assert_eq!(result_trait, result_function);
    }

    #[test]
    fn test_all_container_types_consistency() {
        let input_data = [1.0, 1.0, 1.0, 0.0];
        let expected = [3.0, 1.0, 1.0, -1.0];

        let vec_data = input_data.to_vec();
        let vec_result = vec_data.fwht().unwrap();
        assert_eq!(vec_result, expected.to_vec());

        // Array
        let array_result = input_data.fwht().unwrap();
        assert_eq!(array_result, expected);

        #[cfg(feature = "ndarray")]
        {
            // ndarray
            use ::ndarray::Array1;
            let ndarray_data = Array1::from(input_data.to_vec());
            let ndarray_result = ndarray_data.fwht().unwrap();
            let ndarray_expected = Array1::from(expected.to_vec());
            assert_eq!(ndarray_result, ndarray_expected);
        }
    }

    #[test]
    fn test_different_numeric_types() {
        let data_f32 = vec![1.0f32, 1.0, 1.0, 0.0];
        let result_f32 = data_f32.fwht().unwrap();
        assert_eq!(result_f32, vec![3.0f32, 1.0, 1.0, -1.0]);

        let data_i32 = vec![1i32, 1, 1, 0];
        let result_i32 = data_i32.fwht().unwrap();
        assert_eq!(result_i32, vec![3i32, 1, 1, -1]);
        // i64
        let data_i64 = vec![1i64, 1, 1, 0];
        let result_i64 = data_i64.fwht().unwrap();
        assert_eq!(result_i64, vec![3i64, 1, 1, -1]);
    }

    #[test]
    fn test_power_of_two_validation() {
        let invalid_data = vec![1.0, 2.0, 3.0];

        // Should return error with trait API
        let mut data = invalid_data.clone();
        let result = data.fwht_mut();
        assert!(result.is_err());

        // Should return error with function API
        let mut data = invalid_data.clone();
        let result = fwht_mut(&mut data);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_cases() {
        let empty_vec: Vec<f64> = vec![];
        let empty_result = empty_vec.fwht().unwrap();
        assert_eq!(empty_result.len(), 0);

        let single_vec = vec![42.0];
        let single_result = single_vec.fwht().unwrap();
        assert_eq!(single_result, vec![42.0]);

        let two_array = [3.0, 5.0];
        let two_result = two_array.fwht().unwrap();
        assert_eq!(two_result, [8.0, -2.0]);
    }

    #[test]
    fn test_functional_style() {
        let inputs = &[
            vec![1.0, 0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 1.0, 1.0],
        ];

        let results: Vec<_> = inputs.iter().map(|data| data.fwht().unwrap()).collect();

        let expected = vec![
            vec![2.0, 2.0, 0.0, 0.0],
            vec![2.0, 0.0, 2.0, 0.0],
            vec![3.0, -1.0, -1.0, -1.0],
        ];

        assert_eq!(results, expected);
    }
}
