//! FWHT implementation for ndarray::Array1<T>
//!
//! This module provides the Fast Walsh-Hadamard Transform implementation
//! for `ndarray::Array1<T>` containers when the "ndarray" feature is enabled.

use crate::core::fwht_slice;
use crate::traits::FWHT;
use std::ops::{Add, Sub};

/// Implementation of FWHT for ndarray::Array1<T>
///
/// This implementation works with `ndarray::Array1<T>` where `T` implements
/// the required arithmetic operations. The array must be contiguous in memory
/// for the transform to work.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "ndarray")]
/// # {
/// use fwht::FWHT;
/// use ndarray::Array1;
///
/// let mut data = Array1::from(vec![1.0, 1.0, 1.0, 0.0]);
/// data.fwht_mut().unwrap();
/// let expected = Array1::from(vec![3.0, 1.0, 1.0, -1.0]);
/// assert_eq!(data, expected);
///
/// let data2 = Array1::from(vec![1.0, 1.0, 1.0, 0.0]);
/// let result = data2.fwht().unwrap();
/// let expected = Array1::from(vec![3.0, 1.0, 1.0, -1.0]);
/// assert_eq!(result, expected);
/// # }
/// ```
///
/// # Panics
///
/// Panics if the array is not contiguous in memory.
#[cfg(feature = "ndarray")]
impl<T> FWHT<T> for ndarray::Array1<T>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + Clone,
{
    fn fwht_mut(&mut self) -> Result<(), &'static str> {
        if let Some(slice) = self.as_slice_mut() {
            fwht_slice(slice)
        } else {
            Err("Array must be contiguous for FWHT")
        }
    }

    fn fwht(&self) -> Result<Self, &'static str> {
        let mut result = self.clone();
        result.fwht_mut()?;
        Ok(result)
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_ndarray_fwht_mut() {
        let mut data = Array1::from(vec![1.0, 1.0, 1.0, 0.0]);
        data.fwht_mut().unwrap();
        let expected = Array1::from(vec![3.0, 1.0, 1.0, -1.0]);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_ndarray_fwht_copy() {
        let data = Array1::from(vec![1.0, 1.0, 1.0, 0.0]);
        let result = data.fwht().unwrap();
        let expected = Array1::from(vec![3.0, 1.0, 1.0, -1.0]);
        assert_eq!(result, expected);

        let original_expected = Array1::from(vec![1.0, 1.0, 1.0, 0.0]);
        assert_eq!(data, original_expected);
    }

    #[test]
    fn test_ndarray_fwht_size_8() {
        let mut data = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        data.fwht_mut().unwrap();
        let expected = Array1::from(vec![4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_ndarray_fwht_integers() {
        let data = Array1::from(vec![1i32, 2i32, 3i32, 4i32]);
        let result = data.fwht().unwrap();
        let expected = Array1::from(vec![10i32, -2i32, -4i32, 0i32]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ndarray_fwht_f32() {
        let data = Array1::from(vec![1.0f32, 1.0, 1.0, 0.0]);
        let result = data.fwht().unwrap();
        let expected = Array1::from(vec![3.0f32, 1.0, 1.0, -1.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ndarray_fwht_empty() {
        let mut data = Array1::from(vec![] as Vec<f64>);
        data.fwht_mut().unwrap();
        assert_eq!(data.len(), 0);

        let result = data.fwht().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_ndarray_fwht_single_element() {
        let data = Array1::from(vec![42.0]);
        let result = data.fwht().unwrap();
        let expected = Array1::from(vec![42.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ndarray_fwht_size_2() {
        let data = Array1::from(vec![3.0, 5.0]);
        let result = data.fwht().unwrap();
        let expected = Array1::from(vec![8.0, -2.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ndarray_fwht_non_power_of_two() {
        let mut data = Array1::from(vec![1.0, 2.0, 3.0]);
        let result = data.fwht_mut();
        assert!(result.is_err());
    }

    #[test]
    fn test_ndarray_fwht_involution() {
        let original = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let mut data = original.clone();

        data.fwht_mut().unwrap();
        data.fwht_mut().unwrap();
        let n = data.len() as f64;
        for x in data.iter_mut() {
            *x /= n;
        }

        for (actual, expected) in data.iter().zip(original.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ndarray_from_different_sources() {
        let data1 = Array1::from(vec![1.0, 1.0, 0.0, 0.0]);
        let result1 = data1.fwht().unwrap();
        let expected1 = Array1::from(vec![2.0, 0.0, 2.0, 0.0]);
        assert_eq!(result1, expected1);

        let slice = &[1.0, 1.0, 0.0, 0.0];
        let data2 = Array1::from(slice.to_vec());
        let result2 = data2.fwht().unwrap();
        assert_eq!(result2, expected1);

        let data3: Array1<f64> = Array1::from(
            (0..4)
                .map(|i| if i < 2 { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
        );
        let result3 = data3.fwht().unwrap();
        assert_eq!(result3, expected1);
    }

    #[test]
    fn test_ndarray_size_16() {
        let data = Array1::from(vec![
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ]);
        let result = data.fwht().unwrap();

        // The result should be well-defined for this specific input pattern
        let expected = Array1::from(vec![
            8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        assert_eq!(result, expected);
    }
}
