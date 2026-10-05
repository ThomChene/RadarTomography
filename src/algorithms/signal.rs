use ndarray::{Array1, Array2, NdFloat};
use num_complex::Complex;
use rustfft::{FftNum, FftPlanner};
use rustfft::num_traits::ToPrimitive;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalDomain {
    Time,      // sampling_rate is Hz, start_offset is seconds
    Frequency, // sampling_rate is Hz/bin (resolution), start_offset is Hz
}

#[derive(Debug, Clone, PartialEq)]
pub struct Signal<T:FftNum+NdFloat> {
    /// Single array containing complex numbers
    pub data: Array1<Complex<T>>,
    /// If Time: Sampling frequency (Hz). If Frequency: Frequency resolution (Hz/bin).
    pub sampling_rate: T,
    /// If Time: Start time (seconds). If Frequency: Start frequency (Hz).
    pub start_offset: T,
    pub domain: SignalDomain,
}

impl<T:FftNum+NdFloat> Signal<T> {
    pub fn new(data: Array1<Complex<T>>, sampling_rate: T, start_offset: T, domain: SignalDomain) -> Self {
        Signal {
            data,
            sampling_rate,
            start_offset,
            domain,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Performs Fast Fourier Transform (Time -> Frequency)
    pub fn fft(&self) -> Signal<T> {
        assert_eq!(self.domain, SignalDomain::Time, "FFT requires Time-domain signal");

        let n = self.len();
        let mut buffer = self.data.clone();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(buffer.as_slice_mut().expect("Array must be contiguous"));

        // Frequency resolution: Delta_f = fs / N
        let freq_resolution = self.sampling_rate / T::from_usize(n).unwrap();

        Signal::new(
            buffer,
            freq_resolution,
            T::zero(), // FFT results always start at 0 Hz (DC component)
            SignalDomain::Frequency,
        )
    }

    /// Performs Inverse Fast Fourier Transform (Frequency -> Time)
    pub fn ifft(&self) -> Signal<T> {
        assert_eq!(self.domain, SignalDomain::Frequency, "IFFT requires Frequency-domain signal");

        let n = self.len();
        let mut buffer = self.data.clone();

        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n);
        ifft.process(buffer.as_slice_mut().expect("Array must be contiguous"));

        // rustfft is unnormalized, so we must divide by N
        for val in buffer.iter_mut() {
            *val = *val / T::from_usize(n).unwrap();
        }

        // Sampling frequency
        let fs = T::from_usize(n).unwrap() * self.sampling_rate;

        Signal::new(
            buffer,
            fs,
            T::zero(), // IFFT results start at t=0
            SignalDomain::Time,
        )
    }

    /// Time-domain convolution: (f * g)[n] = sum(f[m] * g[n-m])
    /// Complexity: O(N * M)
    pub fn convolve_direct(&self, other: &Signal<T>) -> Signal<T> {
        assert_eq!(self.domain, SignalDomain::Time);
        assert_eq!(other.domain, SignalDomain::Time);

        let n = self.len();
        let m = other.len();
        let out_len = n + m - 1;

        let mut res_data = Array1::from_elem(out_len, Complex::new(T::zero(), T::zero()));

        for i in 0..n {
            for j in 0..m {
                res_data[i + j] = res_data[i + j] + self.data[i] * other.data[j];
            }
        }

        Signal::new(
            res_data,
            self.sampling_rate,
            self.start_offset + other.start_offset,
            SignalDomain::Time,
        )
    }

    /// FFT-accelerated convolution: F(f * g) = F(f) * F(g)
    /// Complexity: O(N log N)
    pub fn convolve_fft(&self, other: &Signal<T>) -> Signal<T> {
        assert_eq!(self.domain, SignalDomain::Time);
        assert_eq!(other.domain, SignalDomain::Time);

        let n = self.len();
        let m = other.len();
        let out_len = n + m - 1;

        // 1. Pad both signals to out_len and transform to frequency domain
        let s1_freq = self.fft_padded(out_len);
        let s2_freq = other.fft_padded(out_len);

        // 2. Point-wise multiplication in the frequency domain
        let mut res_freq_data = Array1::from_elem(out_len, Complex::new(T::zero(), T::zero()));
        for i in 0..out_len {
            res_freq_data[i] = s1_freq.data[i] * s2_freq.data[i];
        }

        // 3. Transform back to time domain
        let mut res_time = Signal::new(
            res_freq_data,
            s1_freq.sampling_rate,
            T::zero(),
            SignalDomain::Frequency,
        ).ifft();

        res_time.start_offset = self.start_offset + other.start_offset;
        
        res_time
    }

    /// Helper to perform FFT with zero-padding to prevent circular convolution
    fn fft_padded(&self, target_len: usize) -> Signal<T> {
        let mut padded_data = Array1::from_elem(target_len, Complex::new(T::zero(), T::zero()));
        
        // Copy original data into the padded array
        for (i, val) in self.data.iter().enumerate() {
            padded_data[i] = *val;
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(target_len);
        fft.process(padded_data.as_slice_mut().expect("Array must be contiguous"));

        Signal::new(
            padded_data,
            self.sampling_rate / T::from_usize(target_len).unwrap(),
            T::zero(),
            SignalDomain::Frequency,
        )
    }

    // --- Utilities to access parts of the signal ---
    pub fn get_real(&self) -> Array1<T> {
        self.data.map(|c| c.re)
    }

    pub fn get_imag(&self) -> Array1<T> {
        self.data.map(|c| c.im)
    }
}

impl<T:FftNum+NdFloat> Signal<T> {
    /// Creates a Linear Frequency Modulated (LFM) Chirp signal.
    /// 
    /// # Arguments
    /// * `f_start` - Starting frequency in Hz.
    /// * `f_end` - Ending frequency in Hz.
    /// * `duration` - Duration of the chirp in seconds.
    /// * `sampling_rate` - Sampling frequency in Hz.
    /// * `amplitude` - Peak amplitude.
    pub fn create_chirp(
        f_start: T,
        f_end: T,
        num_samples: usize,
        sampling_rate: T,
        amplitude: T,
    ) -> Self {
        let duration = T::from_usize(num_samples).unwrap()/sampling_rate;
        let mut data = Array1::from_elem(num_samples, Complex::new(T::zero(), T::zero()));
        
        let k = (f_end - f_start) / duration; // Chirp rate (Hz/s)
        let pi = T::from_f64(std::f64::consts::PI).unwrap();

        for n in 0..num_samples {
            //let t = T::from_usize(n).unwrap()/sampling_rate - duration/T::from(2.0).unwrap();
            let t = T::from_usize(n).unwrap()/sampling_rate;
            
            // Phase formula for LFM: phi(t) = 2 * pi * (f_start * t + 0.5 * k * t^2)
            let phase = T::from_f64(2.0).unwrap() * pi * (f_start * t +  T::from_f64(0.5).unwrap() * k * t * t);
            
            // We create the complex analytic signal: A * exp(j * phase)
            data[n] = Complex::new(
                amplitude * phase.cos(),
                amplitude * phase.sin()
            );
        }

        Signal::new(
            data,
            sampling_rate,
            T::zero(), // Starts at t=0
            SignalDomain::Time,
        )
    }

    /// Applies a Matched Filter to the current signal using a template.
    /// This performs cross-correlation: y[n] = x[n] ⊛ s*[n]
    /// 
    /// # Arguments
    /// * `template` - The known signal template (the chirp used to transmit).
    /// 
    /// # Returns
    /// A new Signal containing the correlation output.
    pub fn apply_matched_filter(&self, template: &Signal<T>) -> Signal<T> {
        assert_eq!(self.domain, SignalDomain::Time, "Received signal must be in Time domain");
        assert_eq!(template.domain, SignalDomain::Time, "Template must be in Time domain");

        // A Matched Filter is the convolution of the signal with the 
        // time-reversed, complex-conjugated template.
        // In the frequency domain, this is: Y(f) = X(f) * conj(Template(f))
        
        // Step 1: Create a conjugated version of the template
        let mut conj_template_data = template.data.clone();
        for val in conj_template_data.iter_mut() {
            *val = val.conj();
        }
        
        let conj_template = Signal::new(
            conj_template_data,
            template.sampling_rate,
            template.start_offset,
            template.domain,
        );

        // Step 2: Use our existing FFT convolution. 
        // Because we conjugated the template, 'convolve_fft' effectively performs correlation.
        self.convolve_fft(&conj_template)
    }
}

/// Converts raytracing output into a vector of Signals.
pub fn convert_raytracing_to_signals<T>(
    all_amplitudes: Vec<Vec<T>>,
    all_distances: Vec<Vec<T>>,
    sampling_rate: T, 
    speed_of_light: T
) -> Vec<Signal<T>>
where
    T: FftNum + NdFloat + ToPrimitive,
{
    let n_receivers = all_amplitudes.len();
    let mut signals = Vec::with_capacity(n_receivers);

    for r in 0..n_receivers {
        let amplitudes = &all_amplitudes[r];
        let distances = &all_distances[r];

        if amplitudes.is_empty() {
            signals.push(Signal::new(
                Array1::from_elem(1, Complex::new(T::zero(), T::zero())),
                sampling_rate,
                T::zero(),
                SignalDomain::Time,
            ));
            continue; 
        }

        let arrival_times: Vec<T> = distances.iter()
            .map(|&d| d / speed_of_light)
            .collect();

        let t_min = arrival_times.iter().fold(T::max_value(), |a, &b| a.min(b));
        let t_max = arrival_times.iter().fold(T::min_value(), |a, &b| a.max(b));
        
        let duration = t_max - t_min;
        let num_samples = ((duration * sampling_rate).ceil() + T::one()).to_usize().unwrap();

        let mut data = Array1::from_elem(num_samples, Complex::new(T::zero(), T::zero()));

        for (i, &amp) in amplitudes.iter().enumerate() {
            let t_arrival = arrival_times[i];
            let relative_time = t_arrival - t_min;
            let index = (relative_time * sampling_rate).round().to_usize().unwrap();

            if index < num_samples {
                data[index].re += amp;
            }
        }

        signals.push(Signal::new(
            data,
            sampling_rate,
            t_min, 
            SignalDomain::Time,
        ));
    }

    signals
}

pub fn l2_norm<T: NdFloat>(x: &Array1<T>) -> T {
    x.dot(x).sqrt()
}

pub fn scalar_prod<T: NdFloat>(x: &Array1<T>, y: &Array1<T>) -> T {
    x.dot(y)
}

/// Creates a 2D image (Array2) from a slice of signals after applying a matched filter.
/// Each row of the resulting image corresponds to the magnitude of the convolution of the received signal.
pub fn create_sar_image<T: FftNum + NdFloat>(
    signals: &[Signal<T>],
    template: &Signal<T>,
) -> Array2<T> {
    let n_receivers = signals.len();
    
    // 1. Calculate the matched filtered signals to find the required dimensions
    let matched_signals: Vec<Signal<T>> = signals
        .iter()
        .map(|s| s.apply_matched_filter(template))
        .collect();

    let max_len = matched_signals.iter().map(|s| s.len()).max().unwrap_or(0);

    // 2. Build the 2D array
    let mut image = Array2::zeros((n_receivers, max_len));

    for (i, s) in matched_signals.iter().enumerate() {
        let mag = s.data.map(|c| c.norm());
        for (j, &val) in mag.iter().enumerate() {
            image[[i, j]] = val;
        }
    }

    image
}

/// Creates a 2D image (Array2) from a slice of signals.
/// Each row of the resulting image corresponds to the magnitude of the signal.
pub fn create_magnitude_image<T: FftNum + NdFloat>(
    signals: &[Signal<T>],
) -> Array2<T> {
    let n_receivers = signals.len();
    let max_len = signals.iter().map(|s| s.len()).max().unwrap_or(0);
    
    let mut image = Array2::zeros((n_receivers, max_len));

    for (i, s) in signals.iter().enumerate() {
        let mag = s.data.map(|c| c.norm());
        for (j, &val) in mag.iter().enumerate() {
            image[[i, j]] = val;
        }
    }

    image
}
