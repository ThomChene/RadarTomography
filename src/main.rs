use ndarray::arr1;
use radar_and_tomography::map::face::{Receiver, TypeFace};
use radar_and_tomography::map::raytracing;
use radar_and_tomography::map::object;
use radar_and_tomography::map::object::{Object,Transmitter};
use radar_and_tomography::algorithms::signal as signal;

use radar_and_tomography::plot as plot;
use std::f64::consts::PI as Pi;
use indicatif::{ProgressBar, ProgressStyle};

// Todo:
// Check R star
// faire une struct point ou bien vérifier que les valeurs soient en 3D 
// conditions pour lesquelles face est une ligne en fonction du cross product
// add sized to faces?
// look at rayon
// creer nouvelle fonction pour creer les positions du trans
// plot signal reçu, plot matched avec chirp, plot image raw, plot image transformée par SAR si ligne droite, plot tomography sinon

fn main() {
    const SPEED_OF_LIGHT: f64 = 299_792_458.0;
    const SAMPLING_RATE: f64 = 2.0*1e7;
    const F_MAX_CHIRP: f64 = 1e7;
    const N_SAMPLES_CHIRP: usize = (SAMPLING_RATE*3.0/1e5).floor() as usize;
    let n_receiver = 1;
    let n_rays = 100;
    let min_amplitude_ray = 1e-12;

    // 1. Define the terrain bounds
    let min_x = -1000.0;
    let max_x = 1000.0;
    let min_y = -1000.0;
    let max_y = 1000.0;
    let min_z = 0.0;
    let max_z = 0.0;

    let bounds_min = arr1(&[min_x, min_y, min_z]);
    let bounds_max = arr1(&[max_x, max_y, max_z]);

    // 2. Generate the random mountain map
    let mut map = Object::generate_mountain_map(
        15,            // 15 mountains
        bounds_min, 
        bounds_max, 
        100.0,          // min radius
        500.0,          // max radius
        50.0,          // min height
        100.0,          // max height
        12,            // min sides (square-ish)
        48,            // max sides (round cone)
        1.0            // attenuation
    );

    let length_x_transceiver = 2.0;
    let length_y_transceiver = 1.0;
    let wavelength = 0.1;
    let precision = (500,500);
    let theta_transceiver = Pi*4.0/5.0;
    let phi_transceiver = Pi/5.0;

    let max_x_i = 100;
    let max_y_i = 1;
    let min_x = -1000.0;
    let max_x = 1000.0;
    let min_y = 0.0;
    let max_y = 0.0;
    let z = 2000.0;

    let positions = object::get_positions(max_x_i, max_y_i, min_x, max_x, min_y, max_y, z, theta_transceiver, phi_transceiver, length_x_transceiver, length_y_transceiver, wavelength, precision);

    let my_signal = signal::Signal::create_chirp(0.0, F_MAX_CHIRP, N_SAMPLES_CHIRP, SAMPLING_RATE, 1.0);
    // 1. Plot the constellation of a chirp
    plot::plot_signal::plot_signal(&my_signal, "chirp_time.png", plot::plot_signal::PlotMode::RealImag, "Chirp in time");

    // 2. Plot the waveform (Amplitude and Phase)
    plot::plot_signal::plot_signal(&my_signal, "waveform.png", plot::plot_signal::PlotMode::MagnitudePhase, "Time Domain Waveform");

    // 3. Plot the spectrum after FFT
    let spectrum = my_signal.fft();
    plot::plot_signal::plot_signal(&spectrum, "spectrum.png", plot::plot_signal::PlotMode::Spectrum, "Frequency Spectrum");

    // 4. Plot the spectrum after IFFT
    let new_signal = spectrum.ifft();
    plot::plot_signal::plot_signal(&new_signal, "new_waveform.png", plot::plot_signal::PlotMode::RealImag, "Time Domain Waveform");

    let mut all_received_signals = Vec::new();
    let mut all_rays_to_plot = Vec::new();

    let pb = ProgressBar::new(positions.len() as u64);
    pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    for position in positions{
        pb.inc(1);
        let (position_center, phi_theta, position_receiver, beamforming) = position;

        // Create transmitter
        let transmitter = Transmitter::new(beamforming, position_center, phi_theta);

        let receiver = Receiver::new(0);
        let receiver_points = Object::parallelogram_points(position_receiver);

        let receiver_object = Object::new(vec!(receiver_points), vec!(TypeFace::Receiver(receiver)));

        map.push(receiver_object);

        let (all_amplitudes, all_distances, rays_to_plot) = raytracing::raytracing(n_rays, n_receiver, &transmitter, &map, min_amplitude_ray);

        map.pop();
        for rays in rays_to_plot{
            all_rays_to_plot.push(rays);
        }

        let impulsional_response_receivers = signal::convert_raytracing_to_signals(all_amplitudes, all_distances, SAMPLING_RATE, SPEED_OF_LIGHT);

        // Collect the raw signal at this receiver position
        all_received_signals.push(impulsional_response_receivers[0].clone());
    }
    pb.finish_with_message("done");


    // --- SAR Processing ---
    // 1. Generate the SAR image 
    let sar_image = signal::create_sar_image(&all_received_signals, &my_signal);

    // 2. Plot the SAR image
    plot::plot_signal::plot_image(&sar_image, "sar_image.png", "SAR Image (Matched Filter Magnitude)");

    // 3. Plot the actual rays (using the rays you collected)
    plot::plot_beamforming::plot_ray_directions_3d(&all_rays_to_plot, "realized_rays.gif");

    // 4. Plot the map
    //plot::plot_map::plot_map_with_objects(all_rays_to_plot, &map); 
    plot::plot_map::plot_map_with_objects(None, &map); 

    let synthesized_image = radar_and_tomography::algorithms::sar::synthesis(
           &sar_image,
           (max_x - min_x) / f64::from(max_x_i), // step
           2000.0, // length_synthetic
           z, // distance_antenna
           SPEED_OF_LIGHT,
           SAMPLING_RATE,
       );
    plot::plot_signal::plot_image(&synthesized_image, "synthesized_image.png", "Synthesized Image");

}
