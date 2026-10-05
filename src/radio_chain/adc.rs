use ndarray::{Array1, ArrayView1};
use num_complex::Complex;
use ndarray::NdFloat;
use std::marker::PhantomData;
use num_traits::FromPrimitive;

pub struct Adc<T: NdFloat> {
    pub downsampling_factor: usize,
    pub bit_depth: u32,
    _phantom: PhantomData<T>,
}

impl<T: NdFloat + FromPrimitive> Adc<T> {
    pub fn new(downsampling_factor: usize, bit_depth: u32) -> Self {
        Adc { 
            downsampling_factor, 
            bit_depth,
            _phantom: PhantomData,
        }
    }

    /// Performs downsampling and quantization.
    pub fn sample(&self, input: &Array1<Complex<T>>) -> Array1<Complex<T>> {
        let n = input.len();
        let out_len = n / self.downsampling_factor;
        let mut output = Array1::from_elem(out_len, Complex::new(T::zero(), T::zero()));

        let quant_step = T::one() / T::from_f64(2.0f64.powi(self.bit_depth as i32 - 1)).unwrap();

        for i in 0..out_len {
            let idx = i * self.downsampling_factor;
            let val = input[idx];
            
            // Quantization
            let q_re = (val.re / quant_step).round() * quant_step;
            let q_im = (val.im / quant_step).round() * quant_step;
            
            output[i] = Complex::new(q_re, q_im);
        }

        output
    }
}
