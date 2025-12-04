//! Core FWHT algorithm implementation
//!
//! This module contains the fundamental Fast Walsh-Hadamard Transform algorithm
//! that operates on slices. All other implementations build upon this core function.

use std::ops::{Add, Sub};

/// Core FWHT algorithm that operates on mutable slices
///
/// This is the fundamental implementation of the Fast Walsh-Hadamard Transform.
/// It modifies the input slice in-place using the standard butterfly operations.
///
/// # Requirements
///
/// - The slice length must be a power of 2
/// - Elements must implement `Add`, `Sub`, and `Copy`
///
/// # Errors
///
/// Returns an error if the input length is not a power of 2.
///
/// # Examples
///
/// ```
/// use fwht::core::fwht_slice;
///
/// let mut data = [1.0, 1.0, 1.0, 0.0];
/// fwht_slice(&mut data).unwrap();
/// assert_eq!(data, [3.0, 1.0, 1.0, -1.0]);
/// ```
pub fn fwht_slice<T>(data: &mut [T]) -> Result<(), &'static str>
where
    T: Add<Output = T> + Sub<Output = T> + Copy,
{
    let n = data.len();

    if n == 0 {
        return Ok(());
    }

    if !n.is_power_of_two() {
        return Err("Input length must be a power of 2");
    }

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }

    Ok(())
}

/// Validates that a length is suitable for FWHT
///
/// Returns `true` if the length is a power of 2 (including 0 and 1).
pub fn is_valid_fwht_length(n: usize) -> bool {
    n == 0 || n.is_power_of_two()
}

/// Computes the next power of 2 greater than or equal to n
///
/// This can be useful for zero-padding data to a valid FWHT length.
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        n.next_power_of_two()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_slice_basic() {
        let mut data = [1f64, 1f64, 1f64, 0f64];
        fwht_slice(&mut data).unwrap();
        assert_eq!(data, [3f64, 1f64, 1f64, -1f64]);
    }

    #[test]
    fn test_fwht_slice_size_8() {
        let mut data = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        fwht_slice(&mut data).unwrap();
        assert_eq!(data, [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_fwht_slice_single_element() {
        let mut data = [42.0];
        fwht_slice(&mut data).unwrap();
        assert_eq!(data, [42.0]);
    }

    #[test]
    fn test_fwht_slice_two_elements() {
        let mut data = [3.0, 5.0];
        fwht_slice(&mut data).unwrap();
        assert_eq!(data, [8.0, -2.0]);
    }

    #[test]
    fn test_fwht_slice_empty() {
        let mut data: [f64; 0] = [];
        fwht_slice(&mut data).unwrap();
    }

    #[test]
    fn test_fwht_slice_integers() {
        let mut data = [1i32, 2i32, 3i32, 4i32];
        fwht_slice(&mut data).unwrap();
        assert_eq!(data, [10i32, -2i32, -4i32, 0i32]);
    }

    #[test]
    fn test_fwht_slice_involution_property() {
        let original = [1.0, 2.0, 3.0, 4.0];
        let mut data = original;

        fwht_slice(&mut data).unwrap();
        fwht_slice(&mut data).unwrap();

        let n = data.len() as f64;
        for x in &mut data {
            *x /= n;
        }

        for (actual, expected) in data.iter().zip(original.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fwht_slice_non_power_of_two() {
        let mut data = [1.0, 2.0, 3.0];
        let result = fwht_slice(&mut data);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Input length must be a power of 2");
    }

    #[test]
    fn test_is_valid_fwht_length() {
        assert!(is_valid_fwht_length(0));
        assert!(is_valid_fwht_length(1));
        assert!(is_valid_fwht_length(2));
        assert!(is_valid_fwht_length(4));
        assert!(is_valid_fwht_length(8));
        assert!(is_valid_fwht_length(16));

        assert!(!is_valid_fwht_length(3));
        assert!(!is_valid_fwht_length(5));
        assert!(!is_valid_fwht_length(6));
        assert!(!is_valid_fwht_length(7));
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(4), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(9), 16);
    }
}
