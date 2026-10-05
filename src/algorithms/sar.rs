use ndarray::Array2;

pub fn synthesis(
    raw_image: &Array2<f64>,
    step: f64,
    length_synthetic: f64,
    distance_antenna: f64,
    speed_of_light: f64,
    sampling_rate: f64,
) -> Array2<f64> {
    let (rows, cols) = raw_image.dim();
    let step_j = speed_of_light / sampling_rate;
    let mut image_synthesised = Array2::<f64>::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            let n_start = (-length_synthetic / (2.0 * step)).trunc() as i32;
            let n_end = (length_synthetic / (2.0 * step)).trunc() as i32;

            for n in n_start..n_end {
                let new_i = i as i32 + n;
                let new_j = j as i32 + ((n as f64 * step).powi(2) / (2.0 * distance_antenna * step_j)).trunc() as i32;

                if new_i > 0 && (new_i as usize) < rows && new_j < (cols as i32) {
                    sum += raw_image[[new_i as usize, new_j as usize]];
                }
            }
            image_synthesised[[i, j]] = sum;
        }
    }
    image_synthesised
}
