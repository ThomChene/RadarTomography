use ndarray::Array1;
use num_complex::Complex;
use ndarray::NdFloat;

pub struct Lna<T: NdFloat> {
    pub gain: T,
    pub noise_figure: T, // In dB
}

impl<T: NdFloat> Lna<T> {
    pub fn new(gain: T, noise_figure: T) -> Self {
        Lna { gain, noise_figure }
    }

    /// Applies gain to the signal.
    pub fn amplify(&self, input: &Array1<Complex<T>>) -> Array1<Complex<T>> {
        input.map(|c| c * self.gain)
    }
}
