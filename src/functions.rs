//! Function-based FWHT API
//!
//! This module provides the function-based API that works with any type
//! implementing `AsMut<[T]>` and `Clone`.

use crate::core::fwht_slice;
use std::ops::{Add, Sub};

/// Apply FWHT in-place to any container that can provide a mutable slice
///
/// This function-based API works with any type implementing `AsMut<[T]>`.
///
/// # Examples
///
/// ```
/// use fwht::fwht_mut;
///
/// let mut data = vec![1.0, 1.0, 1.0, 0.0];
/// fwht_mut(&mut data).unwrap();
/// assert_eq!(data, vec![3.0, 1.0, 1.0, -1.0]);
///
/// let mut array = [1.0, 1.0, 1.0, 0.0];
/// fwht_mut(&mut array).unwrap();
/// assert_eq!(array, [3.0, 1.0, 1.0, -1.0]);
/// ```
///
/// # Errors
///
/// Returns an error if the container length is not a power of 2.
pub fn fwht_mut<T, V>(data: &mut T) -> Result<(), &'static str>
where
    T: AsMut<[V]> + ?Sized,
    V: Add<Output = V> + Sub<Output = V> + Copy,
{
    fwht_slice(data.as_mut())
}

/// Apply FWHT and return a new container with the result
///
/// This function creates a copy of the input and applies FWHT to the copy.
///
/// # Examples
///
/// ```
/// use fwht::fwht;
///
/// let data = vec![1.0, 1.0, 1.0, 0.0];
/// let result = fwht(&data).unwrap();
/// assert_eq!(result, vec![3.0, 1.0, 1.0, -1.0]);
/// assert_eq!(data, vec![1.0, 1.0, 1.0, 0.0]); // Original unchanged
///
/// let array = [1.0, 1.0, 1.0, 0.0];
/// let result = fwht(&array).unwrap();
/// assert_eq!(result, [3.0, 1.0, 1.0, -1.0]);
/// ```
///
/// # Errors
///
/// Returns an error if the container length is not a power of 2.
pub fn fwht<T, V>(data: &T) -> Result<T, &'static str>
where
    T: Clone + AsMut<[V]>,
    V: Add<Output = V> + Sub<Output = V> + Copy,
{
    let mut result = data.clone();
    fwht_mut(&mut result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_mut_vec() {
        let mut data = vec![1f64, 1f64, 1f64, 0f64];
        fwht_mut(&mut data).unwrap();
        assert_eq!(data, vec![3f64, 1f64, 1f64, -1f64]);
    }

    #[test]
    fn test_fwht_copy_vec() {
        let data = vec![1f64, 1f64, 1f64, 0f64];
        let result = fwht(&data).unwrap();
        assert_eq!(result, vec![3f64, 1f64, 1f64, -1f64]);
        assert_eq!(data, vec![1f64, 1f64, 1f64, 0f64]);
    }

    #[test]
    fn test_fwht_mut_array() {
        let mut data = [1f64, 1f64, 1f64, 0f64];
        fwht_mut(&mut data).unwrap();
        assert_eq!(data, [3f64, 1f64, 1f64, -1f64]);
    }

    #[test]
    fn test_fwht_copy_array() {
        let data = [1f64, 1f64, 1f64, 0f64];
        let result = fwht(&data).unwrap();
        assert_eq!(result, [3f64, 1f64, 1f64, -1f64]);
        assert_eq!(data, [1f64, 1f64, 1f64, 0f64]);
    }

    #[test]
    fn test_fwht_slice() {
        let mut buffer = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let slice = &mut buffer[0..4];
        fwht_mut(slice).unwrap();
        assert_eq!(slice, &mut [3.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_fwht_size_8() {
        let mut data = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        fwht_mut(&mut data).unwrap();
        assert_eq!(data, vec![4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_fwht_integers() {
        let data = vec![1i32, 2i32, 3i32, 4i32];
        let result = fwht(&data).unwrap();
        assert_eq!(result, vec![10i32, -2i32, -4i32, 0i32]);
    }

    #[test]
    fn test_fwht_empty() {
        let mut data: Vec<f64> = vec![];
        fwht_mut(&mut data).unwrap();
        assert_eq!(data.len(), 0);

        let result = fwht(&data).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_fwht_non_power_of_two() {
        let mut data = vec![1.0, 2.0, 3.0];
        let result = fwht_mut(&mut data);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Input length must be a power of 2");
    }

    #[test]
    fn test_fwht_with_different_containers() {
        let vec_data = vec![1.0, 1.0, 0.0, 0.0];
        let vec_result = fwht(&vec_data).unwrap();
        assert_eq!(vec_result, vec![2.0, 0.0, 2.0, 0.0]);

        let array_data = [1.0, 1.0, 0.0, 0.0];
        let array_result = fwht(&array_data).unwrap();
        assert_eq!(array_result, [2.0, 0.0, 2.0, 0.0]);

        assert_eq!(vec_result, array_result.to_vec());
    }

    #[test]
    fn test_api_consistency_with_trait() {
        use crate::traits::FWHT;

        let data = vec![1.0, 2.0, 3.0, 4.0];

        let result_function = fwht(&data).unwrap();
        let result_trait = data.fwht().unwrap();

        assert_eq!(result_function, result_trait);
    }
}
