use std::f64::consts::PI as Pi_64;

use ndarray::NdFloat;

fn sinc<T:NdFloat>(x:T)->Result<T,T>{
    if x==T::zero(){
        Ok(T::one())
    }
    else{
        let pi_opt = T::from(Pi_64);
        match pi_opt{
            Some(pi) => Ok(T::sin(pi*x)/(pi*x)),
            None => Err(T::zero()),
        }
    }
}

pub struct Beamforming<T>
{
    beam_pattern: Vec<Vec<T>>, //Matrix containing the intensity of the beam in a spheric coordinates system (phi and theta)
    n_phi: u32,
    n_theta: u32,
    min_phi: T,
    max_phi: T,
    min_theta: T,
    max_theta: T,
    max_abs_value_pattern: T,
}

impl<T:NdFloat> Beamforming<T>{
    /// Creates a matrix with the intensity of the beam. It is in spheric coordinates(radians).
    /// For a rectangular antenna with dimensions L and l the beam shape in the direction normal to the antenna is in sinc.
    /// The pattern is only computed in a relevant region around the main lobe.
    pub fn new_rectangular_antenna(l_x:T, l_y:T, wavelength:T, precision:(u32,u32))->Self{
        let pi = T::from(Pi_64).unwrap();
        // Min values and Max values around the main lobe
        let min_phi = -pi;
        let max_phi = pi;
        let min_theta = T::zero();
        let max_theta = T::min(pi, T::from(3.0).unwrap()*pi*wavelength/T::min(l_x,l_y));
        let mut max_abs_value_pattern = T::min_value();

        let (max_row, max_col) = precision;
        let mut beam_pattern = Vec::new();
        for row in 0..max_row{
            let mut vec_col = Vec::new();
            for col in 0..max_col{
                let phi_radian = (max_phi-min_phi)*T::from(col).unwrap()/T::from(max_col).unwrap() + min_phi;
                // For a uniform distribution on the sphere
                let theta_radian = (max_theta-min_theta)*T::acos(T::from(2*row).unwrap()/T::from(max_row).unwrap() - T::one())/pi+ min_theta;

                let x = T::sin(theta_radian)*T::cos(phi_radian);
                let y = T::sin(theta_radian)*T::sin(phi_radian);

                let sinc_x = sinc(l_x*x/wavelength).unwrap();
                let sinc_y = sinc(l_y*y/wavelength).unwrap();
                if max_abs_value_pattern<T::abs(sinc_x*sinc_y){
                    max_abs_value_pattern = T::abs(sinc_x*sinc_y);
                    }
                vec_col.push(T::abs(sinc_x*sinc_y));
            }
            beam_pattern.push(vec_col);
        }

        let beamforming = Beamforming
        {
            beam_pattern,
            n_theta: max_row,
            n_phi: max_col,
            min_phi,
            max_phi,
            min_theta,
            max_theta,
            max_abs_value_pattern,
        };
        beamforming
    }

    pub fn get_pattern(&self)->&Vec<Vec<T>>{
        &self.beam_pattern
    }

    pub fn get_values(&self)->(u32,u32,T,T,T,T,T){
        (self.n_phi, self.n_theta, self.min_phi, self.max_phi, self.min_theta, self.max_theta, self.max_abs_value_pattern)
    }
}