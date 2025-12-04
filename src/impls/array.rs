//! FWHT implementation for static arrays [T; N]
//!
//! This module provides the Fast Walsh-Hadamard Transform implementation
//! for static arrays of fixed size.

use crate::core::fwht_slice;
use crate::traits::FWHT;
use std::ops::{Add, Sub};

/// Implementation of FWHT for static arrays [T; N]
///
/// This implementation works with any fixed-size array `[T; N]` where `T` implements
/// the required arithmetic operations. Static arrays have the advantage of being
/// stack-allocated and having their size known at compile time.
///
/// # Examples
///
/// ```
/// use fwht::FWHT;
///
/// let mut data = [1.0, 1.0, 1.0, 0.0];
/// data.fwht_mut().unwrap();
/// assert_eq!(data, [3.0, 1.0, 1.0, -1.0]);
///
/// let data2 = [1.0, 1.0, 1.0, 0.0];
/// let result = data2.fwht().unwrap();
/// assert_eq!(result, [3.0, 1.0, 1.0, -1.0]);
/// ```
impl<T, const N: usize> FWHT<T> for [T; N]
where
    T: Add<Output = T> + Sub<Output = T> + Copy,
{
    fn fwht_mut(&mut self) -> Result<(), &'static str> {
        fwht_slice(self.as_mut_slice())
    }

    fn fwht(&self) -> Result<Self, &'static str> {
        let mut result = *self;
        result.fwht_mut()?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_fwht_mut() {
        let mut data = [1f64, 1f64, 1f64, 0f64];
        data.fwht_mut().unwrap();
        assert_eq!(data, [3f64, 1f64, 1f64, -1f64]);
    }

    #[test]
    fn test_array_fwht_copy() {
        let data = [1f64, 1f64, 1f64, 0f64];
        let result = data.fwht().unwrap();
        assert_eq!(result, [3f64, 1f64, 1f64, -1f64]);
        assert_eq!(data, [1f64, 1f64, 1f64, 0f64]);
    }

    #[test]
    fn test_array_fwht_size_8() {
        let mut data = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        data.fwht_mut().unwrap();
        assert_eq!(data, [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_array_fwht_integers() {
        let data = [1i32, 2i32, 3i32, 4i32];
        let result = data.fwht().unwrap();
        assert_eq!(result, [10i32, -2i32, -4i32, 0i32]);
    }

    #[test]
    fn test_array_fwht_f32() {
        let data = [1.0f32, 1.0, 1.0, 0.0];
        let result = data.fwht().unwrap();
        assert_eq!(result, [3.0f32, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_array_fwht_size_1() {
        let mut data = [42.0];
        data.fwht_mut().unwrap();
        assert_eq!(data, [42.0]);
    }

    #[test]
    fn test_array_fwht_size_2() {
        let data = [3.0, 5.0];
        let result = data.fwht().unwrap();
        assert_eq!(result, [8.0, -2.0]);
    }

    #[test]
    fn test_array_fwht_size_16() {
        let mut data = [
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        data.fwht_mut().unwrap();

        let expected = [
            8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(data, expected);
    }

    #[test]
    fn test_array_fwht_non_power_of_two() {
        let mut data = [1.0, 2.0, 3.0];
        let result = data.fwht_mut();
        assert!(result.is_err());
    }

    #[test]
    fn test_array_fwht_involution() {
        let original = [1.0, 2.0, 3.0, 4.0];
        let mut data = original;

        data.fwht_mut().unwrap();
        data.fwht_mut().unwrap();

        let n = data.len() as f64;
        for x in &mut data {
            *x /= n;
        }

        for (actual, expected) in data.iter().zip(original.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_array_zero_length() {
        let mut data: [f64; 0] = [];
        data.fwht_mut().unwrap();

        let result = data.fwht().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_array_different_numeric_types() {
        let data_i8 = [1i8, 1, 1, 0];
        let result_i8 = data_i8.fwht().unwrap();
        assert_eq!(result_i8, [3i8, 1, 1, -1]);

        let data_i16 = [1i16, 1, 1, 0];
        let result_i16 = data_i16.fwht().unwrap();
        assert_eq!(result_i16, [3i16, 1, 1, -1]);

        let data_i64 = [1i64, 1, 1, 0];
        let result_i64 = data_i64.fwht().unwrap();
        assert_eq!(result_i64, [3i64, 1, 1, -1]);
    }
}
