#![deny(missing_docs)]

//!# Kallman
//!
//! This crate implements simple Kallman Filtering.
//! Feature set is limited but this is considered for usage in object tracking
//! so this is all that's needed.
//!
//!```
//!# use assert_approx_eq::assert_approx_eq;
//!
//! use kallman::{KallmanFilterBuilder, KallmanState, nalgebra};
//! use nalgebra::{SMatrix, SVector};
//!
//! let initial_state = nalgebra::vector![0.0, 1.0];
//! let initial_covariance = nalgebra::matrix![
//!     1.0, 0.0;
//!     0.0, 1.0;
//! ];
//!
//! let control_input = nalgebra::vector![2.0f64, 1.0];
//! let process_noise_covariance = nalgebra::matrix![2.0, 1.0; 0.0, 0.5];
//!
//! let kalman_filter = KallmanFilterBuilder::new()
//!     .f::<()>(nalgebra::matrix![1.0, 1.0; 0.0, 1.0])
//!     .q::<()>(process_noise_covariance)
//!     .h::<()>(nalgebra::matrix![1.0, 0.0; 0.0, 1.0])
//!     .w::<()>(control_input)
//!     .build::<()>();
//!
//! let mut kf: KallmanState<f64, 2, 2> = kalman_filter.init(initial_state, initial_covariance);
//!
//! kf.predict();
//!
//! assert_approx_eq!(kf.state().x, 3.0, 1e-5);
//! assert_approx_eq!(kf.state().y, 2.0, 1e-5);
//!
//! assert_approx_eq!(kf.covariance().m11, 4.0);
//! assert_approx_eq!(kf.covariance().m12, 2.);
//! assert_approx_eq!(kf.covariance().m21, 1.0);
//! assert_approx_eq!(kf.covariance().m22, 1.5);
//!
//!```

use std::{
    ops::{AddAssign, MulAssign},
    sync::Arc,
};

/// Helpers for initializing filter and working with it's results.
///
/// There is no point in providing wrappers for matrix operations, so why just
/// not export it.
///
/// TODO: Maybe use simpler slices
pub use nalgebra;
use nalgebra::{SMatrix, SVector};

use num_traits::{One, Zero};

/// Description of a number type needed to use this crate
pub trait Number:
    Copy
    + Clone
    + nalgebra::ComplexField
    + From<f32>
    + Default
    + Send
    + Sync
    + std::fmt::Debug
    + std::fmt::Display
    + 'static
    + PartialEq
    + Zero
    + One
    + AddAssign
    + MulAssign
{
}

impl Number for f64 {}
impl Number for f32 {}
impl Number for nalgebra::Complex<f32> {}

/// This is the core of the crate but it only holds data shared between all of
/// it's children's objects.
#[derive(Debug, Clone, PartialEq)]
pub struct KallmanFilter<T: Number, const S: usize, const O: usize> {
    f: SMatrix<T, S, S>,
    ft: SMatrix<T, S, S>,
    h: SMatrix<T, O, S>,
    ht: SMatrix<T, S, O>,
    q: SMatrix<T, S, S>,
    w: SVector<T, S>,
}

impl<T: Number, const SS: usize, const OS: usize> Default for KallmanFilter<T, SS, OS> {
    fn default() -> Self {
        let f = SMatrix::zeros();
        let ft = SMatrix::zeros();
        let h = SMatrix::zeros();
        let ht = SMatrix::zeros();
        let q = SMatrix::zeros();
        let w = SVector::zeros();

        Self { f, ft, h, ht, q, w }
    }
}

/// Since [`KallmanFilterBuilder`] uses type system to assert that all the matrices
/// have been initialized, this type is neeeded to check if a condition is met.
pub trait IsTrue<const F: bool> {}

/// Since [`KallmanFilterBuilder`] uses type system to assert that all the matrices
/// have been initialized, this type is neeeded to check if a condition is met.
pub trait IsFalse<const F: bool> {}

impl<T> IsTrue<true> for T {}
impl<T> IsFalse<false> for T {}

/// This struct is used for creating [`KallmanFilter`], and makes sure it has
/// been properly initialized
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KallmanFilterBuilder<
    T: Number,
    const S: usize,
    const O: usize,
    const F: bool,
    const H: bool,
    const Q: bool,
    const W: bool,
> {
    filter: KallmanFilter<T, S, O>,
}

impl<T: Number, const S: usize, const O: usize>
    KallmanFilterBuilder<T, S, O, false, false, false, false>
{
    /// Initializes new empty builder object
    pub fn new() -> Self {
        Default::default()
    }
}

impl<
        T: Number,
        const S: usize,
        const O: usize,
        const F: bool,
        const H: bool,
        const Q: bool,
        const W: bool,
    > KallmanFilterBuilder<T, S, O, F, H, Q, W>
{
    /// Finishes initialization of Kallman Filter.
    /// This function will compile only if all four fields have been initialized.
    pub fn build<C>(self) -> Arc<KallmanFilter<T, S, O>>
    where
        C: IsTrue<F> + IsTrue<H> + IsTrue<Q> + IsTrue<W>,
    {
        Arc::new(self.filter)
    }

    /// Sets F and F^T matrices and can only be called once per [`KallmanFilterBuilder`]
    /// object
    pub fn f<C: IsFalse<F>>(
        mut self,
        f: SMatrix<T, S, S>,
    ) -> KallmanFilterBuilder<T, S, O, true, H, Q, W> {
        self.filter.f = f;
        self.filter.ft = self.filter.f.transpose();

        KallmanFilterBuilder::<T, S, O, true, H, Q, W> {
            filter: self.filter,
        }
    }

    /// Sets H and H^T matrices and can only be called once per [`KallmanFilterBuilder`]
    /// object
    pub fn h<C: IsFalse<H>>(
        mut self,
        h: SMatrix<T, O, S>,
    ) -> KallmanFilterBuilder<T, S, O, F, true, Q, W> {
        self.filter.h = h;
        self.filter.ht = self.filter.h.transpose();

        KallmanFilterBuilder::<T, S, O, F, true, Q, W> {
            filter: self.filter,
        }
    }

    /// Sets Q matrix. Like [`Self::f`] and [`Self::h`] it can be called only
    /// once per [`KallmanFilterBuilder`] object.
    pub fn q<C: IsFalse<Q>>(
        mut self,
        q: SMatrix<T, S, S>,
    ) -> KallmanFilterBuilder<T, S, O, F, H, true, W> {
        self.filter.q = q;

        KallmanFilterBuilder::<T, S, O, F, H, true, W> {
            filter: self.filter,
        }
    }

    /// Sets w matrix. Like [`Self::f`] and [`Self::h`] it can be called only
    /// once per [`KallmanFilterBuilder`] object.
    pub fn w<C: IsFalse<W>>(
        mut self,
        w: SVector<T, S>,
    ) -> KallmanFilterBuilder<T, S, O, F, H, Q, true> {
        self.filter.w = w;

        KallmanFilterBuilder::<T, S, O, F, H, Q, true> {
            filter: self.filter,
        }
    }
}

/// This structure represents a single object that is Kallman filtered.
/// It simply holds it's state and uncertainty, while referencing a [`KallmanFilter`]
/// object
pub struct KallmanState<T: Number, const S: usize, const O: usize> {
    x: SVector<T, S>,
    p: SMatrix<T, S, S>,
    filter: Arc<KallmanFilter<T, S, O>>,
}

impl<T: Number, const S: usize, const O: usize> KallmanState<T, S, O> {
    /// Predicts next position of the filter
    pub fn predict(&mut self) {
        self.x = self.filter.f * self.x + self.filter.w;
        self.p = self.filter.f * self.p * self.filter.ft + self.filter.q;
    }

    /// Gets the value of state
    pub fn state(&self) -> &SVector<T, S> {
        &self.x
    }

    /// Gets the value of the covariance
    pub fn covariance(&self) -> &SMatrix<T, S, S> {
        &self.p
    }

    /// Updates current state with the supplied observation and confidence.
    ///
    /// This function will change and is intended for object tracking.
    pub fn update(&mut self, z: SVector<T, O>, confidence: f64) {
        let r = SMatrix::<T, O, O>::from_diagonal(&nalgebra::SVector::<T, O>::from_element(
            <f32 as Into<T>>::into(confidence.powf(2.0) as f32),
        ));

        let sinv = {
            let mut s: SMatrix<T, O, O> = self.filter.h * self.p * self.filter.ht + r;
            s.try_inverse_mut();
            s
        };

        let k = self.p * self.filter.ht * sinv;

        self.x = (SMatrix::identity() - k * self.filter.h) * self.x + k * z;
        self.p = (SMatrix::identity() - k * self.filter.h) * self.p;
    }

    /// Updates current state with the supplied observation.
    ///
    /// It is different from [`Self::update`] because value of observation
    /// covariance matrix is directly supplied.
    pub fn update_r(&mut self, z: SVector<T, O>, r: SMatrix<T, O, O>) {
        let sinv = {
            let mut s: SMatrix<T, O, O> = self.filter.h * self.p * self.filter.ht + r;
            s.try_inverse_mut();
            s
        };

        let k = self.p * self.filter.ht * sinv;

        self.x = (SMatrix::identity() - k * self.filter.h) * self.x + k * z;
        self.p = (SMatrix::identity() - k * self.filter.h) * self.p;
        eprintln!("X: {}, P: {}", self.x, self.p);
    }
}

impl<T: Number, const S: usize, const O: usize> KallmanFilter<T, S, O> {
    /// Creates new [`KallmanState`] referencing `self`.
    pub fn init(
        self: &Arc<Self>,
        intial_state: SVector<T, S>,
        initial_covariance: SMatrix<T, S, S>,
    ) -> KallmanState<T, S, O> {
        KallmanState {
            x: intial_state,
            p: initial_covariance,
            filter: Arc::clone(self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn init() {
        let kalman_filter = KallmanFilterBuilder::new()
            .f::<()>(SMatrix::from_element(1.0))
            .q::<()>(SMatrix::from_element(2.0))
            .h::<()>(SMatrix::from_element(1.0))
            .w::<()>(SVector::from_element(0.0))
            .build::<()>();

        let initial_state = nalgebra::vector![0.0, 0.0];
        let initial_covariance = nalgebra::matrix![
            100.0, 0.0;
            0.0, 100.0;
        ];

        let kf: KallmanState<f64, 2, 2> = kalman_filter.init(initial_state, initial_covariance);

        assert_approx_eq!(kf.state().x, initial_state.x);
        assert_approx_eq!(kf.state().y, initial_state.y);

        assert_approx_eq!(kf.covariance().m11, initial_covariance.m11);
        assert_approx_eq!(kf.covariance().m12, initial_covariance.m12);
        assert_approx_eq!(kf.covariance().m21, initial_covariance.m21);
        assert_approx_eq!(kf.covariance().m22, initial_covariance.m22);
    }

    #[test]
    fn operation() {
        let initial_state = nalgebra::vector![0.0, 1.0];
        let initial_covariance = nalgebra::matrix![
            1.0, 0.0;
            0.0, 1.0;
        ];

        let control_input = nalgebra::vector![2.0f64, 1.0];
        let process_noise_covariance = nalgebra::matrix![2.0, 1.0; 0.0, 0.5];

        let kalman_filter = KallmanFilterBuilder::new()
            .f::<()>(nalgebra::matrix![1.0, 1.0; 0.0, 1.0])
            .q::<()>(process_noise_covariance)
            .h::<()>(nalgebra::matrix![1.0, 0.0; 0.0, 1.0])
            .w::<()>(control_input)
            .build::<()>();

        let mut kf: KallmanState<f64, 2, 2> = kalman_filter.init(initial_state, initial_covariance);

        kf.predict();

        assert_approx_eq!(kf.state().x, 3.0, 1e-5);
        assert_approx_eq!(kf.state().y, 2.0, 1e-5);
        assert_approx_eq!(kf.covariance().m11, 4.0);
        assert_approx_eq!(kf.covariance().m12, 2.);
        assert_approx_eq!(kf.covariance().m21, 1.0);
        assert_approx_eq!(kf.covariance().m22, 1.5);

        let z = nalgebra::vector![1.0f64, 0.7];
        let r = nalgebra::matrix![2.0, 0.0; 0.0, 0.5];

        kf.update_r(z, r);

        assert_approx_eq!(kf.state().x, 1.28);
        assert_approx_eq!(kf.state().y, 0.99, 1e-6);

        assert_approx_eq!(kf.covariance().m11, 1.2);
        assert_approx_eq!(kf.covariance().m12, 0.2);
        assert_approx_eq!(kf.covariance().m21, 0.1);
        assert_approx_eq!(kf.covariance().m22, 0.35);
    }
}
