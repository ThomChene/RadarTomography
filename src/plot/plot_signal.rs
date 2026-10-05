use plotters::prelude::*;
use crate::algorithms::signal::{Signal, SignalDomain};
use ndarray::NdFloat;
use rustfft::FftNum;
use num_traits::ToPrimitive;

pub enum PlotMode {
    /// Two stacked subplots: Top = Real part, Bottom = Imaginary part (vs Time/Freq)
    RealImag,
    /// Dual-axis line plot: X=Time/Freq, Y1=Amplitude, Y2=Phase
    MagnitudePhase,
    /// Magnitude spectrum in Decibels (dB)
    Spectrum,
}

pub fn plot_signal<T>(
    signal: &Signal<T>,
    filename: &str,
    mode: PlotMode,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> 
where 
    T: NdFloat + FftNum
{
    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let n = signal.len();
    let is_freq = signal.domain == SignalDomain::Frequency;

    // --- 1. CALCULATE RAW X-VALUES AND DETERMINE SCALING ---
    // We calculate the "true" value in base units (seconds or Hz) first
    let raw_x: Vec<f64> = (0..n).map(|i| {
        let val = if is_freq {
            // In Frequency domain, sampling_rate is the frequency resolution (Hz/bin)
            signal.start_offset.to_f64().unwrap() + (T::from_usize(i).unwrap() * signal.sampling_rate).to_f64().unwrap()
        } else {
            // In Time domain, sampling_rate is the sampling frequency (Hz)
            signal.start_offset.to_f64().unwrap() + (T::from_usize(i).unwrap() / signal.sampling_rate).to_f64().unwrap()
        };
        val
    }).collect();

    // Find max value to decide the unit scale
    let max_x_raw = raw_x.iter().fold(0.0, |a: f64, &b| a.max(b));

    // Determine multiplier and suffix
    let (multiplier, unit_label) = if is_freq {
        if max_x_raw < 1e3 {
            (1.0, "Hz")
        } else if max_x_raw < 1e6 {
            (1e3, "kHz")
        } else {
            (1e6, "MHz")
        }
    } else {
        // Time scaling
        if max_x_raw < 1e-3 {
            (1e6, "µs")
        } else if max_x_raw < 1.0 {
            (1e3, "ms")
        } else {
            (1.0, "s")
        }
    };

    // Scale the X values for plotting
    let x_values: Vec<f64> = raw_x.iter().map(|&x| x / multiplier).collect();
    let x_min = x_values[0];
    let x_max = x_values[n - 1];

    let margin = 5;

    match mode {
        PlotMode::RealImag => {
            let real_vals: Vec<f64> = signal.data.iter().map(|c| c.re.to_f64().unwrap()).collect();
            let imag_vals: Vec<f64> = signal.data.iter().map(|c| c.im.to_f64().unwrap()).collect();

            let (upper, lower) = root.split_vertically(768/2);
            
            // 1. TOP SUBPLOT: REAL PART
            let min_re = real_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_re = real_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            let mut real_chart = ChartBuilder::on(&upper)
                .margin(margin)
                .set_all_label_area_size(50)
                .caption(format!("{} (Real Part)", title), ("sans-serif", 25))
                .build_cartesian_2d(x_min..x_max, min_re..max_re)?;

            real_chart.configure_mesh()
                .x_desc(unit_label)
                .y_desc("Real")
                .draw()?;

            real_chart.draw_series(LineSeries::new(
                x_values.iter().zip(real_vals.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?;

            // 2. BOTTOM SUBPLOT: IMAGINARY PART
            let min_im = imag_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_im = imag_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let mut imag_chart = ChartBuilder::on(&lower)
                .margin(margin)
                .set_all_label_area_size(50)
                .caption("Imaginary Part", ("sans-serif", 25))
                .build_cartesian_2d(x_min..x_max, min_im..max_im)?;

            imag_chart.configure_mesh()
                .x_desc(unit_label)
                .y_desc("Imag")
                .draw()?;

            imag_chart.draw_series(LineSeries::new(
                x_values.iter().zip(imag_vals.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))?;
        }

        PlotMode::MagnitudePhase => {
            let magnitudes: Vec<f64> = signal.data.iter().map(|c| c.norm().to_f64().unwrap()).collect();
            let phases: Vec<f64> = signal.data.iter().map(|c| c.arg().to_f64().unwrap()).collect();

            let (upper, lower) = root.split_vertically(768/2);

            // 1. TOP CHART: Magnitude
            let min_mag = magnitudes.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_mag = magnitudes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            let mut mag_chart = ChartBuilder::on(&upper)
                .margin(margin)
                .set_all_label_area_size(50)
                .caption(format!("{} (Magnitude)", title), ("sans-serif", 25))
                .build_cartesian_2d(x_min..x_max, min_mag..max_mag)?;

            mag_chart.configure_mesh()
                .x_desc(unit_label)
                .y_desc("Mag")
                .draw()?;

            mag_chart.draw_series(LineSeries::new(
                x_values.iter().zip(magnitudes.iter()).map(|(&x, &y)| (x, y)),
                &BLUE,
            ))?;

            // 2. BOTTOM CHART: Phase
            let mut phase_chart = ChartBuilder::on(&lower)
                .margin(margin)
                .set_all_label_area_size(50)
                .caption("Phase (rad)", ("sans-serif", 25))
                .build_cartesian_2d(x_min..x_max, -std::f64::consts::PI..std::f64::consts::PI)?;

            phase_chart.configure_mesh()
                .x_desc(unit_label)
                .y_desc("Phase")
                .draw()?;

            phase_chart.draw_series(LineSeries::new(
                x_values.iter().zip(phases.iter()).map(|(&x, &y)| (x, y)),
                &RED,
            ))?;
        }

        PlotMode::Spectrum => {
            let db_values: Vec<f64> = signal.data.iter()
                .map(|c| c.norm().to_f64().unwrap())
                .collect();

            let min_val = db_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = db_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let mut chart = ChartBuilder::on(&root)
                .margin(margin)
                .set_all_label_area_size(50)
                .caption(title, ("sans-serif", 30))
                .build_cartesian_2d(x_min..x_max, min_val..max_val)?;

            chart.configure_mesh()
                .x_desc(unit_label)
                .y_desc("Magnitude")
                .draw()?;

            chart.draw_series(LineSeries::new(
                x_values.iter().zip(db_values.iter()).map(|(&x, &y)| (x, y)),
                &GREEN,
            ))?;
        }
    }

    root.present()?;
    Ok(())
}

pub fn plot_image<T>(
    image: &ndarray::Array2<T>,
    filename: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    T: ndarray::NdFloat + ToPrimitive,
{
    let (rows, cols) = image.dim();
    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .set_all_label_area_size(50)
        .caption(title, ("sans-serif", 30))
        .build_cartesian_2d(0..cols, 0..rows)?;

    chart.configure_mesh()
        .x_desc("Time / Range index")
        .y_desc("Receiver index")
        .draw()?;

    // Find min and max for color scaling
    let mut min_val = T::max_value();
    let mut max_val = T::min_value();
    for val in image.iter() {
        if *val < min_val { min_val = *val; }
        if *val > max_val { max_val = *val; }
    }

    let min_f = min_val.to_f64().unwrap();
    let max_f = max_val.to_f64().unwrap();

    for r in 0..rows {
        for c in 0..cols {
            let val = image[[r, c]].to_f64().unwrap();
            let norm_val = if max_f > min_f { (val - min_f) / (max_f - min_f) } else { 0.0 };
            
            // Simple color mapping: Blue for low, Red for high
            let color = RGBColor((255.0 * (1.0 - norm_val)) as u8, 0, (255.0 * norm_val) as u8);
            
            chart.draw_series(std::iter::once(Rectangle::new(
                [(c, r), (c + 1, r + 1)],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}
