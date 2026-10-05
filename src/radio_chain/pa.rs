use ndarray::Array1;
use num_complex::Complex;
use ndarray::NdFloat;

pub struct PowerAmplifier<T: NdFloat> {
    pub gain: T,
    pub saturation_level: T,
}

impl<T: NdFloat> PowerAmplifier<T> {
    pub fn new(gain: T, saturation_level: T) -> Self {
        PowerAmplifier { gain, saturation_level }
    }

    /// Applies gain and then clips the signal if it exceeds the saturation level.
    pub fn amplify(&self, input: &Array1<Complex<T>>) -> Array1<Complex<T>> {
        let mut output = input.map(|c| c * self.gain);

        for val in output.iter_mut() {
            let mag = val.norm();
            if mag > self.saturation_level {
                let scale = self.saturation_level / mag;
                *val = *val * scale;
            }
        }

        output
    }
}
