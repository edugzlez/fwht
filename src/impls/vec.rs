//! FWHT implementation for Vec<T>
//!
//! This module provides the Fast Walsh-Hadamard Transform implementation
//! for `Vec<T>` containers.

use crate::core::fwht_slice;
use crate::traits::FWHT;
use std::ops::{Add, Sub};

/// Implementation of FWHT for Vec<T>
///
/// This implementation works with any `Vec<T>` where `T` implements
/// the required arithmetic operations.
///
/// # Examples
///
/// ```
/// use fwht::FWHT;
///
/// let mut data = vec![1.0, 1.0, 1.0, 0.0];
/// data.fwht_mut().unwrap();
/// assert_eq!(data, vec![3.0, 1.0, 1.0, -1.0]);
///
/// let data2 = vec![1.0, 1.0, 1.0, 0.0];
/// let result = data2.fwht().unwrap();
/// assert_eq!(result, vec![3.0, 1.0, 1.0, -1.0]);
/// ```
impl<T> FWHT<T> for Vec<T>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + Clone,
{
    fn fwht_mut(&mut self) -> Result<(), &'static str> {
        fwht_slice(self.as_mut_slice())
    }

    fn fwht(&self) -> Result<Self, &'static str> {
        let mut result = self.clone();
        result.fwht_mut()?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_fwht_mut() {
        let mut data = vec![1f64, 1f64, 1f64, 0f64];
        data.fwht_mut().unwrap();
        assert_eq!(data, vec![3f64, 1f64, 1f64, -1f64]);
    }

    #[test]
    fn test_vec_fwht_copy() {
        let data = vec![1f64, 1f64, 1f64, 0f64];
        let result = data.fwht().unwrap();
        assert_eq!(result, vec![3f64, 1f64, 1f64, -1f64]);
        assert_eq!(data, vec![1f64, 1f64, 1f64, 0f64]);
    }

    #[test]
    fn test_vec_fwht_size_8() {
        let mut data = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        data.fwht_mut().unwrap();
        assert_eq!(data, vec![4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vec_fwht_integers() {
        let data = vec![1i32, 2i32, 3i32, 4i32];
        let result = data.fwht().unwrap();
        assert_eq!(result, vec![10i32, -2i32, -4i32, 0i32]);
    }

    #[test]
    fn test_vec_fwht_f32() {
        let data = vec![1.0f32, 1.0, 1.0, 0.0];
        let result = data.fwht().unwrap();
        assert_eq!(result, vec![3.0f32, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_vec_fwht_empty() {
        let mut data: Vec<f64> = vec![];
        data.fwht_mut().unwrap();
        assert_eq!(data.len(), 0);

        let result = data.fwht().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_vec_fwht_single_element() {
        let data = vec![42.0];
        let result = data.fwht().unwrap();
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_vec_fwht_non_power_of_two() {
        let mut data = vec![1.0, 2.0, 3.0];
        let result = data.fwht_mut();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Input length must be a power of 2");
    }

    #[test]
    fn test_vec_fwht_involution() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = original.clone();

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
}
