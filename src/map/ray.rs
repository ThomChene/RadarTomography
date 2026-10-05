use ndarray::{Array1,NdFloat};
use std::fmt;
use crate::algorithms::signal;
use super::face::Face;

pub struct Ray<T:NdFloat>{
    amplitude:T,
    distance:T,
    origin:Array1<T>,
    direction:Array1<T>,
    min_amplitude_ray: T,
}

/// Display:
/// 
impl<T:NdFloat> fmt::Display for Ray<T>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Ray with amplitude {} that crossed {} meters, with origin and direction:\n",&self.amplitude,&self.distance)?;
        write!(f, "origin: {}\n", &self.origin)?;
        write!(f, "direction: {}\n", &self.direction)
    }
}
impl<T:NdFloat> Ray<T>{
    pub fn get_values(&self)->(T, T, &Array1<T>, &Array1<T>, T){
        (self.amplitude, self.distance, &self.origin, &self.direction, self.min_amplitude_ray)
    }
}

/// Example:
///```
/// let amplitude = 1;
/// let distance 0;
/// let origin = arr1(&[0.0,0.0,0.0]);
/// let direction = arr1(&[1.0,0.0,0.0]);
/// let ray = Ray::new(amplitude,distance,origin,direction);
/// println!("{ray}");
///```
/// 
impl<T:NdFloat> Ray<T>{
    pub fn new(amplitude: T,distance: T,origin: Array1<T>,direction: Array1<T>,min_amplitude_ray: T)->Self{
        Ray{
            amplitude,
            distance,
            origin,
            direction,
            min_amplitude_ray,
        }
    }

    /// Returns new reflected rays that will later be propagated or no rays if there are no objects in the direction of the ray,
    /// and also an option that corresponds to:
    /// 1) The amplitude and distance of the ray when it touches a receiver 
    /// 2) None else
    pub fn propagate(self,faces: &Vec<&Face<T>>)->(Vec<Ray<T>>,Option<(T,T,usize)>)
    {
        let mut closest_dist = T::max_value();
        let mut closest_face: Option<&Face<T>> = None;
        let mut closest_intersection_face_ray = None;
        let mut intersection_face_ray; 
        for face in faces.iter(){
            let (normal, offset, p1, p2, p3, squared_norm_first_vector, squared_norm_second_vector, scalar_product_vectors) = face.get_values();
            // First compute the intersection between the ray and the plane
            let new_offset = offset + signal::scalar_prod(normal,&self.origin);
            let other_offset = signal::scalar_prod(normal,&self.direction);
            // If the normal is perpendicular to the direction of the ray, there is no intersection
            if !other_offset.is_zero(){
                let a = -(new_offset/other_offset);
                intersection_face_ray = &self.direction*a + &self.origin;
                // Checks that the intersection is in the direction of propagation of the ray and not behind the transmitter or too closed
                if T::from(1e-2).unwrap()<a{
                    // Then check if the intersection is inside the triangle which means intersection_face_ray-P1 = lambda1*(P2-P1)+lambda2*(P3-P1), with 0<lambda1+lambda2<1, and 0<lambda1<1 and and 0<lambda2<1
                    // By computing <intersection_face_ray-P1,P2-P1> and <intersection_face_ray-P1,P3-P1> we obtain a system, and by solving the system we find lambda1, lambda2 and check that 0<lambda1+lambda2<1 and 0<lambda1<1 and and 0<lambda2<1
                    // Or just 0<lambda1, 0<lambda2 and lambda1+lambda2<1
                    let v1 = &intersection_face_ray-p1;
                    let v2 = p2-p1;
                    let v3 = p3-p1;
                    let sc1 = signal::scalar_prod(&v1,&v2);
                    let sc2 = signal::scalar_prod(&v1,&v3);
                    let s1 = squared_norm_first_vector;
                    let s2 = squared_norm_second_vector;
                    let sp = scalar_product_vectors;
                    let normalization = s2*s1 - sp*sp;
                    let lambda1 = sc1*s2 - sc2*sp;
                    let lambda2 = sc2*s1 - sc1*sp;

                    if T::zero()<lambda1 && T::zero()<lambda2 && lambda1+lambda2<normalization{
                        let diff = &self.origin - &intersection_face_ray;
                        let dist = signal::l2_norm(&diff);
                        if closest_dist>dist{
                            closest_dist = dist;
                            closest_face = Some(face);
                            closest_intersection_face_ray = Some(intersection_face_ray);
                            }
                        }
                    }
                }
            }
        match (closest_face, closest_intersection_face_ray){
            (Some(face), Some(inter)) => face.ray_received(self, closest_dist, inter),
            //None => (vec!(Ray::new(T::zero(), self.distance, self.origin, self.direction)), None),
            _ => (Vec::new(), None),
        }
    }
}
