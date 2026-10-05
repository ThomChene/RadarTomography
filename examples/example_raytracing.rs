use ndarray::{arr1, arr2};
use radar_and_tomography::algorithms::beamforming::Beamforming;
use radar_and_tomography::map::face::{Receiver, TypeFace};
use radar_and_tomography::map::raytracing;
use radar_and_tomography::map::object::{Object,Transmitter};
use radar_and_tomography::algorithms::signal as signal;

use radar_and_tomography::plot as plot;
use std::f64::consts::PI as Pi;

fn main() {
    ////////////////////////
    // General Parameters //
    ////////////////////////
    const SPEED_OF_LIGHT: f64 = 299_792_458.0;
    const F_MAX_CHIRP: f64 = 1e7;
    const SAMPLING_RATE: f64 = 2.0*1e7;
    const N_SAMPLES_CHIRP: usize = (SAMPLING_RATE*3.0/1e5).floor() as usize;
    let n_receiver = 1;
    let n_rays = 100;
    let min_amplitude_ray = 1e-12;

    ////////////////////////
    // Antenna Parameters //
    ////////////////////////
    let length_x_transceiver = 2.0;
    let length_y_transceiver = 1.0;
    let wavelength = 0.1;
    let precision = (500,500);
    let theta_transceiver = Pi*4.0/5.0; // directions theta and phi of the antenna system in spherical coordinates
    let phi_transceiver = Pi/5.0; 
    let beamforming = Beamforming::new_rectangular_antenna(length_x_transceiver, length_y_transceiver, wavelength, precision);
    plot::plot_beamforming::plot_beam_pattern_3d(&beamforming, "beam_pattern.gif"); // Plot the shape of the beam

    ////////////////////
    // Map Parameters //
    ////////////////////
    let min_x = -1000.0;
    let max_x = 1000.0;
    let min_y = -1000.0;
    let max_y = 1000.0;
    let min_z = 0.0;
    let max_z = 0.0;
    let bounds_min = arr1(&[min_x, min_y, min_z]);
    let bounds_max = arr1(&[max_x, max_y, max_z]);

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

    //////////////////////////
    // Position transceiver //
    //////////////////////////
    let max_x_i = 1;
    let max_y_i = 1;
    let min_x = 0.0;
    let max_x = 200.0;
    let min_y = 0.0;
    let max_y = 200.0;
    let z = 2000.0;

    positions.push((position_center, (phi_transceiver, theta_transceiver), vec!(position_bottom_left, position_bottom_right, position_top_right), beamforming));
    let (position_center, phi_theta, position_receiver, beamforming) = position;

    // Create transmitter and receiver
    let transmitter = Transmitter::new(beamforming, position_center, phi_theta);
    let receiver = Receiver::new(0);
    let receiver_points = Object::parallelogram_points(position_receiver);
    let receiver_object = Object::new(vec!(receiver_points), vec!(TypeFace::Receiver(receiver)));
    map.push(receiver_object);

    ////////////////////////
    // Signal transceiver //
    ////////////////////////
    let chirp = signal::Signal::create_chirp(0.0, F_MAX_CHIRP, N_SAMPLES_CHIRP, SAMPLING_RATE, 1.0);

    // Plot the chirp
    plot::plot_signal::plot_signal(&chirp, "chirp_time_real_imag.png", plot::plot_signal::PlotMode::RealImag, "Chirp in time, real and imaginary part");
    plot::plot_signal::plot_signal(&chirp, "chirp_time_mag_phase.png", plot::plot_signal::PlotMode::MagnitudePhase, "Chirp in time, magnitude and phase");
    
    // Plot the spectrum after FFT
    let spectrum = chirp.fft();
    plot::plot_signal::plot_signal(&spectrum, "chirp_spectrum.png", plot::plot_signal::PlotMode::Spectrum, "Spectrum chirp");
    
    // Plot the chirp after IFFT
    let new_signal = spectrum.ifft();
    plot::plot_signal::plot_signal(&new_signal, "chirp_after_ifft.png", plot::plot_signal::PlotMode::RealImag, "Chirp in time, real and imaginary part after fft");

    ////////////////
    // Raytracing //
    ////////////////
    let (all_amplitudes, all_distances, rays_to_plot) = raytracing::raytracing(n_rays, n_receiver, &transmitter, &map, min_amplitude_ray);

    /////////////////////
    // Received signal //
    /////////////////////
    let impulsional_response_receivers = signal::convert_raytracing_to_signals(all_amplitudes, all_distances, SAMPLING_RATE, SPEED_OF_LIGHT);



    // 3. Plot the actual rays (using the rays you collected)
    plot::plot_beamforming::plot_ray_directions_3d(&all_rays_to_plot, "realized_rays.gif");

    // 4. Plot the map
    plot::plot_map::plot_map_with_objects(all_rays_to_plot, &map);
}

//corriger object 
//tester example
//voir example radio