//! Trait definitions for FWHT operations
//!
//! This module defines the core trait that provides a uniform interface
//! for Fast Walsh-Hadamard Transform operations across different container types.

/// Trait for types that support Fast Walsh-Hadamard Transform operations
///
/// This trait provides a uniform interface for applying FWHT to different
/// container types like `Vec<T>`, `[T; N]`, and `ndarray::Array1<T>`.
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
/// assert_eq!(data2, vec![1.0, 1.0, 1.0, 0.0]);
/// ```
pub trait FWHT<T> {
    /// Apply FWHT in-place, modifying the container's data
    ///
    /// This method is more memory efficient as it doesn't create copies.
    /// The container's data is transformed directly.
    ///
    /// # Errors
    ///
    /// Returns an error if the container length is not a power of 2.
    fn fwht_mut(&mut self) -> Result<(), &'static str>;

    /// Apply FWHT and return a new container with the result
    ///
    /// This method preserves the original data by creating a copy
    /// and applying the transform to the copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the container length is not a power of 2.
    fn fwht(&self) -> Result<Self, &'static str>
    where
        Self: Sized;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_trait_signature() {
        fn _test_fwht_mut<C: FWHT<f64>>(container: &mut C) -> Result<(), &'static str> {
            container.fwht_mut()
        }

        fn _test_fwht<C: FWHT<f64>>(container: &C) -> Result<C, &'static str> {
            container.fwht()
        }
    }
}
