use ndarray::{Array1, ArrayView1};
use num_complex::Complex;
use ndarray::NdFloat;
use std::marker::PhantomData;

pub struct Dac<T: NdFloat> {
    pub upsampling_factor: usize,
    _phantom: PhantomData<T>,
}

impl<T: NdFloat> Dac<T> {
    pub fn new(upsampling_factor: usize) -> Self {
        Dac { 
            upsampling_factor,
            _phantom: PhantomData,
        }
    }

    /// Performs simple upsampling by inserting zeros between samples.
    /// In a real DAC, this would be followed by an interpolation filter.
    pub fn upsample(&self, input: &Array1<Complex<T>>) -> Array1<Complex<T>> {
        let n = input.len();
        let mut output = Array1::from_elem(n * self.upsampling_factor, Complex::new(T::zero(), T::zero()));

        for i in 0..n {
            output[i * self.upsampling_factor] = input[i];
        }

        output
    }
}
