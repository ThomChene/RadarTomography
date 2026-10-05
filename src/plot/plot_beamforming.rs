use plotters::prelude::*;
use ndarray::NdFloat;
use crate::algorithms::beamforming::Beamforming;
use rustfft::num_traits::{FromPrimitive, ToPrimitive};
use std::f64::consts::PI as Pi;

/// Plots the theoretical beam pattern as a 3D shape.
pub fn plot_beam_pattern_3d<T: NdFloat + ToPrimitive + FromPrimitive>(
    beamforming: &Beamforming<T>,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {

    let root = BitMapBackend::gif(filename, (1024, 768), 100)?.into_drawing_area();

    let (n_phi, n_theta, min_phi, max_phi, min_theta, max_theta, max_abs) = 
        beamforming.get_values();

    let pattern = beamforming.get_pattern();

    let n_pitch = 1;
    for pitch in 0..n_pitch {
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Theoretical Beam Radiation Pattern ", ("sans-serif", 30))
            //.build_cartesian_3d(min_phi.to_f64().unwrap()..max_phi.to_f64().unwrap(), -100.0..0.0, min_theta.to_f64().unwrap()..max_theta.to_f64().unwrap())?;
            .build_cartesian_3d(-1.0..1.0, -1.0..1.0, -1.0..1.0)?;

        chart.with_projection(|mut p| {
            p.yaw = Pi/4.0 + 2.0 * Pi * f64::from(pitch) / f64::from(n_pitch);
            p.scale = 0.7;
            p.into_matrix()
        });

        chart.configure_axes().draw()?;

        for row in 0..n_theta as usize {
            for col in 0..n_phi as usize {
                let col_norm = T::from(col).unwrap();
                let max_col = T::from(n_phi).unwrap();
                let row_norm = T::from(row).unwrap();
                let max_row = T::from(n_theta).unwrap();

                let min_value = -50.0; // minimum value for the plot

                let phi_radian = ((max_phi - min_phi) * col_norm / max_col + min_phi).to_f64().unwrap();
                let theta_radian = (max_theta - min_theta).to_f64().unwrap()*f64::acos(2.0*(row_norm/max_row).to_f64().unwrap() - 1.0)/Pi + min_theta.to_f64().unwrap();
                let r = 10.0*f64::log((pattern[row][col]/max_abs).to_f64().unwrap(),10.0);

                let r_min = f64::max(min_value,r)/(-min_value) + 1.0; // normalize values between 0 and 1 for the plot

                // Convert Spherical to Cartesians
                let x: f64 = r_min * theta_radian.sin() * phi_radian.cos();
                let y: f64 = r_min * theta_radian.sin() * phi_radian.sin();
                let z: f64 = r_min * theta_radian.cos() ;

                let color = VulcanoHSL::get_color(r_min);
                chart.draw_series(std::iter::once(Circle::new((x.to_f64().unwrap(), z.to_f64().unwrap(), y.to_f64().unwrap()), 2, color.filled())))?;
                //chart.draw_series(std::iter::once(Circle::new((phi_radian,r,theta_radian), 2, color.filled())))?;
            }
        }
        root.present()?;
    }
    println!("Result has been saved to {}", filename);
    Ok(())
}


/// This visualizes the path of the rays as lines in 3D space.
pub fn plot_ray_directions_3d<T: NdFloat + ToPrimitive>(
    rays_segments: &Vec<Vec<(T, T, T)>>, // Changed input type
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {

    let root = BitMapBackend::gif(filename, (1024, 768), 100)?.into_drawing_area();

    let n_pitch = 1;
    for pitch in 0..n_pitch {
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Real Beam Radiation ", ("sans-serif", 30))
            //.build_cartesian_3d(min_phi.to_f64().unwrap()..max_phi.to_f64().unwrap(), -100.0..0.0, min_theta.to_f64().unwrap()..max_theta.to_f64().unwrap())?;
            .build_cartesian_3d(-1.0..1.0, -1.0..1.0, -1.0..1.0)?;

        chart.with_projection(|mut p| {
            p.yaw = Pi/4.0 + 2.0 * Pi * f64::from(pitch) / f64::from(n_pitch);
            p.scale = 0.7;
            p.into_matrix()
        });

        chart.configure_axes().draw()?;

        // Plot each segment as a LineSeries
        for segment in rays_segments {
            // A segment in your raytracing output is a Vec containing [start_point, end_point]
            if segment.len() >= 2 {
                let p1 = segment[0];
                let p2 = segment[1];

                let x = p2.0.to_f64().unwrap() - p1.0.to_f64().unwrap();
                let y = p2.1.to_f64().unwrap() - p1.1.to_f64().unwrap();
                let z = p2.2.to_f64().unwrap() - p1.2.to_f64().unwrap();

                let norm = f64::powf(f64::powi(x , 2) + f64::powi(y , 2) + f64::powi(z , 2), 0.5);

                let x = x/norm;
                let y = y/norm;
                let z = z/norm;

                let color = RED;
                chart.draw_series(std::iter::once(Circle::new((x, z, y), 2, color.filled())))?;
            }
        }
        root.present()?;
    }
    Ok(())
}
