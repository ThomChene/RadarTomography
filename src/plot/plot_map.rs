use ndarray::Array1;
use plotters::prelude::*;
use std::f64::consts::PI as Pi;

use crate::map::object::Object;

const OUT_FILE_NAME: &str = "./map.gif";

/// Helper to convert ndarray Array1 to a tuple with Y and Z swapped
/// Original: (x, y, z) -> New: (x, z, y)
fn arr_to_swapped_tuple(arr: &Array1<f64>) -> (f64, f64, f64) {
    (arr[0], arr[2], arr[1])
}

/// Helper to convert a ray point tuple with Y and Z swapped
/// Original: (x, y, z) -> New: (x, z, y)
fn tuple_to_swapped_tuple(t: (f64, f64, f64)) -> (f64, f64, f64) {
    (t.0, t.2, t.1)
}

pub fn plot_map_with_objects(
    rays: Option<&[Vec<(f64, f64, f64)>]>, 
    objects: &[Object<f64>]
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::gif(OUT_FILE_NAME, (600, 400), 100)?.into_drawing_area();

    let n_pitch = 1;
    for pitch in 0..n_pitch {
        root.fill(&WHITE)?;

        // --- DIMENSION SWAP LOGIC ---
        // Y and Z should be swapped
        let mut chart = ChartBuilder::on(&root)
            .caption("Map ", ("sans-serif", 20))
            .build_cartesian_3d(-1000.0..1000.0, 0.0..700.0, -1000.0..1000.0)?;

        chart.with_projection(|mut p| {
            p.yaw = Pi/4.0 + 2.0 * Pi * f64::from(pitch) / f64::from(n_pitch);
            p.scale = 0.7;
            p.into_matrix()
        });

        chart
            .configure_axes()
            .light_grid_style(BLACK.mix(0.15))
            .max_light_lines(3)
            .draw()?;

        // --- 1. DRAW THE TOWN (Wireframe) ---
        for object in objects {
            for group_faces in object.get_group_faces() {
                for face in group_faces.get_faces() {
                    let (_, _, p1_arr, p2_arr, p3_arr, _, _, _) = face.get_values();
                    
                    let p1 = arr_to_swapped_tuple(p1_arr);
                    let p2 = arr_to_swapped_tuple(p2_arr);
                    let p3 = arr_to_swapped_tuple(p3_arr);

                    let color = RGBColor(100, 100, 100).mix(0.5); 

                    chart.draw_series(LineSeries::new(vec![p1, p2], &color))?;
                    chart.draw_series(LineSeries::new(vec![p2, p3], &color))?;
                    chart.draw_series(LineSeries::new(vec![p3, p1], &color))?;
                }
            }
        }

        // --- 2. DRAW THE RAYS ---
        if let Some(rays_to_draw) = rays {
            for ray in rays_to_draw {
                let swapped_ray: Vec<(f64, f64, f64)> = ray
                    .iter()
                    .map(|&p| tuple_to_swapped_tuple(p))
                    .collect();

                chart.draw_series(LineSeries::new(
                    swapped_ray,
                    &RED,
                ))?;
            }
        }

        root.present()?;
    }

    println!("Result has been saved to {}", OUT_FILE_NAME);
    Ok(())
}