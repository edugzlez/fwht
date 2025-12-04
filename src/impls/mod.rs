//! Implementation modules for different container types
//!
//! This module contains the FWHT trait implementations for various
//! container types like Vec, arrays, and ndarray.

pub mod array;
pub mod vec;

#[cfg(feature = "ndarray")]
pub mod ndarray;
