use ndarray::{arr1, arr2, Array1, Array2, NdFloat};
use std::f64::consts::PI;
use std::fmt;
use rand::{RngExt};

use crate::algorithms::beamforming::Beamforming;
use crate::map::face::Neutral;
use super::ray::Ray;
use super::face::{Face, GroupFaces, TypeFace};
/// An `Object` contains one or multiple `GroupFaces`, and each `GroupFaces` contains one or multiple `Face` that have the same `TypeFace`.
/// An `Object` can move either by translation or rotation. Every point constituting the `Object` will move following the same motion. When points are moved, faces will be updated accordingly.
/// An `Object` contains one or multiple `GroupPoints`, even though each face contains points as well which is redundant, each `Face` is self sufficient and no extra information is required to run raytracing on it.
/// 
pub struct Object<T:NdFloat>{
    group_faces: Vec<GroupFaces<T>>,
    group_points: Vec<GroupPoints<T>>,
}

impl<T:NdFloat> Object<T>{
    pub fn get_group_faces(&self)->&Vec<GroupFaces<T>>{
        &self.group_faces
    }
}
/// `GroupPoint`
pub struct GroupPoints<T:NdFloat>{
    points: Vec<Array1<T>>,
    indices: Array2<usize>,
}

impl<T:NdFloat> GroupPoints<T>{
    pub fn get_point(&self,i:usize)->&Array1<T>{
        &self.points[i]
    }
    pub fn get_indices(&self,n: usize)->(usize,usize,usize){
        (self.indices[[n,0]],self.indices[[n,1]],self.indices[[n,2]])
    }
    pub fn get_number_faces(&self)->usize{
        self.indices.dim().0
    }
    fn move_points(&mut self, vector: &Array1<T>){
        for mut point in self.points.iter(){
            point = &(vector + point);
        }
    }
}

pub fn from_objects_to_faces<T:NdFloat>(objects: &Vec<Object<T>>)->Vec<&Face<T>>{
    let mut vec_faces = Vec::new();
    for object in objects.iter(){
        for group_faces in object.group_faces.iter(){
            vec_faces.extend(group_faces.get_faces());
        }
    }
    vec_faces
}

/// `Transmitter`
pub struct Transmitter<T:NdFloat>{
    beamforming: Beamforming<T>,
    origin_rays: Array1<T>,
    phi_theta_center_direction_ray: (T,T),
}

impl<T:NdFloat> Transmitter<T>{
    pub fn new(beamforming: Beamforming<T>, origin_rays: Array1<T>, phi_theta_center_direction_ray: (T,T))->Self{
        Transmitter{
            beamforming,
            origin_rays,
            phi_theta_center_direction_ray,
        }
    }
    /// Accept/Reject sampling to generate rays according to the beam pattern 
    pub fn create_rays(&self, n_rays: u32, min_amplitude_ray: T)->Vec<Ray<T>>{
        // Rays are centered around this direction
        let (phi_center, theta_center) = self.phi_theta_center_direction_ray;

        // Generate randomly rays around the center
        let mut rng = rand::rng();

        let pattern = self.beamforming.get_pattern();
        let (n_phi, n_theta, min_phi, max_phi, min_theta, max_theta, max_abs_value_pattern) = self.beamforming.get_values();
        let n_unif = 1000;

        let mut n_accepted_rays = 0;
        let mut rays = Vec::new();
        while n_accepted_rays<n_rays {
            let rand_int_phi = rng.random_range(0..n_phi);
            let rand_int_theta = rng.random_range(0..n_theta);

            let rand_unif = T::from(rng.random_range(0..n_unif)).unwrap()/T ::from(n_unif).unwrap();
            let rand_phi = T::from(rand_int_phi).unwrap()/T::from(n_phi).unwrap();
            let rand_theta = T::from(rand_int_theta).unwrap()/T::from(n_theta).unwrap();

            if rand_unif < T::abs(pattern[rand_int_theta as usize][rand_int_phi as usize])/max_abs_value_pattern{
                // Accepts the ray / else rejects it
                let theta_radian = min_theta + (max_theta-min_theta)*rand_theta + theta_center;
                let phi_radian = min_phi + (max_phi-min_phi)*rand_phi + phi_center;

                let direction = arr1(&[T::zero(),T::zero(),T::one()]);

                let mat_rotation_y = arr2(&[
                    [theta_radian.cos(),T::zero(),theta_radian.sin()],
                    [T::zero(),T::one(),T::zero()],
                    [-theta_radian.sin(),T::zero(),theta_radian.cos()]]);
                let mat_rotation_z = arr2(&[
                    [phi_radian.cos(),-phi_radian.sin(),T::zero()],
                    [phi_radian.sin(),phi_radian.cos(),T::zero()],
                    [T::zero(),T::zero(),T::one()]]);

                let direction = direction.dot(&mat_rotation_y.t()).dot(&mat_rotation_z.t());
                let (vec_direction,_) = direction.into_raw_vec_and_offset();
                let direction = arr1(&vec_direction);

                let ray = Ray::new(T::one(), T::zero(), self.origin_rays.clone(), direction, min_amplitude_ray);

                rays.push(ray);
                n_accepted_rays = n_accepted_rays + 1;
            }
        }
        rays
    }
}

/// Display:
/// 
impl<'a,T:NdFloat> fmt::Display for Object<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "This Object with faces:\n")?;
        for group_face in &self.group_faces{
            let vec_faces = group_face.get_faces();
            for v in vec_faces.iter() {
                write!(f, "p1: {}\n", v)?;
            }
        }
        write!(f, "")
    }
}

impl<'a,T:NdFloat> Object<T>{
    pub fn new(group_points: Vec<GroupPoints<T>>, type_face: Vec<TypeFace<T>>)->Self{
        let mut group_faces = Vec::new();
        for (i,group_point) in group_points.iter().enumerate(){
            let group_face = GroupFaces::new(group_point, type_face[i]);
            group_faces.push(group_face);
        }
        let object = Object{
            group_faces,
            group_points,
        };
        object
    }

    pub fn move_object(&mut self,vector:&Array1<T>){
        for group_point in self.group_points.iter_mut(){
            group_point.move_points(vector);
        }
    }

    /// Builds a parallelogram from 3 points
    pub fn parallelogram_points(mut points: Vec<Array1<T>>)->GroupPoints<T>{
        assert!(points.len()==3);
        points.push(&points[1]-&points[0]+&points[2]);
        let indices = arr2(&[[0,1,3],[3,2,0]]);
        GroupPoints{
            points,
            indices,
        }
    }

    /// Builds a parallelepiped from 4 points
    pub fn parallelepiped_points(mut points: Vec<Array1<T>>)->GroupPoints<T>{
        points.push(&points[1]-&points[0]+&points[2]);
        points.push(&points[1]-&points[0]+&points[3]);
        points.push(&points[2]-&points[0]+&points[5]);
        points.push(&points[2]-&points[0]+&points[3]);

        let indices = arr2(&[
            [0,1,4],[4,2,0],
            [2,4,6],[6,7,2],
            [0,1,5],[5,3,0],
            [3,5,6],[6,7,3],
            [1,5,6],[6,4,1],
            [0,3,7],[7,2,0]]);
        GroupPoints{
            points,
            indices,
        }
    }

    /// Builds a mountain (pyramid/cone) shape.
    /// 
    /// # Arguments
    /// * `center` - The center of the base of the mountain [x, y, z].
    /// * `radius` - The radius of the base of the mountain.
    /// * `height` - The height of the peak above the center.
    /// * `n_sides` - Number of sides (use 4 for a square pyramid, 32+ for a cone/round mountain).
    pub fn mountain_points(center: Array1<T>, radius: T, height: T, n_sides: usize) -> GroupPoints<T> {
        let mut points = Vec::new();
        
        // 1. Add the Peak point
        let peak = center.clone() + &arr1(&[T::zero(), T::zero(), height]);
        points.push(peak); // Index 0 is the peak

        // 2. Add the Base Center point (to triangulate the bottom)
        let base_center = center.clone();
        points.push(base_center); // Index 1 is the base center

        // 3. Add the Perimeter points (the base)
        let n_t: T = T::from(n_sides).unwrap();
        let pi_t = T::from(PI).unwrap();

        for i in 0..n_sides {
            let angle = (T::from(i).unwrap() * (T::from(2).unwrap() * pi_t) / n_t) / T::one();
            
            // Calculate offset using trigonometry
            let dx = radius * angle.sin(); // Using sin/cos for circle
            let dy = radius * angle.cos();
            
            // Note: in your coordinate system, you might want to swap dx/dy 
            // depending on which axis you consider "forward"
            let p = center.clone() + &arr1(&[dx, dy, T::zero()]);
            points.push(p); // Indices 2 to (n_sides + 1) are base perimeter
        }

        // 4. Build Indices for Triangulation
        let mut indices_vec = Vec::new();

        // --- SIDE FACES (connecting peak to base edges) ---
        // Triangle: (Peak, Base_i, Base_i+1)
        for i in 0..n_sides {
            let next = (i + 1) % n_sides;
            // Peak is index 0
            // Base points start at index 2
            indices_vec.push(0);
            indices_vec.push(i + 2);
            indices_vec.push(next + 2);
        }

        // --- BASE FACES (connecting base center to base edges) ---
        // Triangle: (Base_Center, Base_i, Base_i+1)
        for i in 0..n_sides {
            let next = (i + 1) % n_sides;
            // Base Center is index 1
            indices_vec.push(1);
            indices_vec.push(i + 2);
            indices_vec.push(next + 2);
        }

        // Flatten indices into the Array2 format required by GroupPoints
        let total_faces = n_sides * 2;
        let mut indices_array = Array2::<usize>::zeros((total_faces, 3));
        for (idx, &val) in indices_vec.iter().enumerate() {
            indices_array[[idx / 3, idx % 3]] = val;
        }

        GroupPoints {
            points,
            indices: indices_array,
        }
    }
}

/// To create maps
/// 
/// 
/// 
impl<T:NdFloat> Object<T>{
    /// Generates a list of random buildings (parallelepipeds) within a specified region.
    pub fn generate_random_flat_town(n_objects: usize, bounds_min: Array1<T>, bounds_max: Array1<T>, min_dim: T, max_dim: T) -> Vec<Object<T>> {
        let mut rng = rand::rng();
        let mut town_objects = Vec::new();

        let p1 = arr1(&[bounds_min[0],bounds_min[1],T::zero()]);
        let p2 = arr1(&[bounds_max[0],bounds_min[1],T::zero()]);
        let p3 = arr1(&[bounds_min[0],bounds_max[1],T::zero()]);

        let neutral = Neutral::new(T::one());
        let neutral_points = Object::parallelogram_points(vec!(p1,p2,p3));
        let ground = Object::new(vec!(neutral_points), vec!(TypeFace::Neutral(neutral)));
        town_objects.push(ground);

        for _ in 0..n_objects {
            // 1. Generate random center position within bounds
            let x_center = T::from(rng.random_range(bounds_min[0].to_f64().unwrap()..(bounds_max[0].to_f64().unwrap()))).unwrap();
            let y_center = T::from(rng.random_range(bounds_min[1].to_f64().unwrap()..(bounds_max[1].to_f64().unwrap()))).unwrap();
            let z_center = T::zero();
            let center = arr1(&[x_center, y_center, z_center]);

            // 2. Generate random dimensions
            let dx = T::from(rng.random_range(min_dim.to_f64().unwrap()..max_dim.to_f64().unwrap())).unwrap();
            let dy = T::from(rng.random_range(min_dim.to_f64().unwrap()..max_dim.to_f64().unwrap())).unwrap();
            let dz = T::from(rng.random_range(min_dim.to_f64().unwrap()..max_dim.to_f64().unwrap())).unwrap();

            // 3. Define the 4 corners of the base (to build the parallelepiped)
            let p0 = center.clone() - &arr1(&[dx/T::from(2.0).unwrap(), dy/T::from(2.0).unwrap(), T::zero()]);
            let p1 = center.clone() + &arr1(&[dx/T::from(2.0).unwrap(), -dy/T::from(2.0).unwrap(), T::zero()]);
            let p2 = center.clone() + &arr1(&[-dx/T::from(2.0).unwrap(), dy/T::from(2.0).unwrap(), T::zero()]);
            let p3 = center.clone() + &arr1(&[-dx/T::from(2.0).unwrap(), -dy/T::from(2.0).unwrap(), dz]);


            let group_points = Object::parallelepiped_points(vec![p0, p1, p2, p3]);
            let neutral = Neutral::new(T::one());
            let object = Object::new(
                vec![group_points], 
                vec![TypeFace::Neutral(neutral)]
            );

            town_objects.push(object);
        }

        town_objects
    }


    /// Generates a list of random mountains within a specified area.
    /// 
    /// # Arguments
    /// * `n_mountains` - Number of mountains to create.
    /// * `bounds_min` - The minimum (x, y, z) of the terrain (usually z=0).
    /// * `bounds_max` - The maximum (x, y, z) of the terrain.
    /// * `min_radius` / `max_radius` - Size of the mountain base.
    /// * `min_height` / `max_height` - Vertical size of the mountain.
    /// * `min_sides` / `max_sides` - Sharpness (4 = pyramid, 32 = smooth cone).
    /// * `attenuation` - The Neutral face attenuation value.
    pub fn generate_mountain_map(
        n_mountains: usize,
        bounds_min: Array1<T>,
        bounds_max: Array1<T>,
        min_radius: T,
        max_radius: T,
        min_height: T,
        max_height: T,
        min_sides: usize,
        max_sides: usize,
        attenuation: T,
    ) -> Vec<Object<T>> {
        let mut rng = rand::rng();
        let mut map = Vec::new();

        // Flat Ground
        let p1 = arr1(&[bounds_min[0],bounds_min[1],T::zero()]);
        let p2 = arr1(&[bounds_max[0],bounds_min[1],T::zero()]);
        let p3 = arr1(&[bounds_min[0],bounds_max[1],T::zero()]);

        let neutral = Neutral::new(T::one());
        let neutral_points = Object::parallelogram_points(vec!(p1,p2,p3));
        let ground = Object::new(vec!(neutral_points), vec!(TypeFace::Neutral(neutral)));
        map.push(ground);

        for _ in 0..n_mountains {
            // 1. Pick random X and Y within bounds
            // We use f64 for RNG and convert to T to satisfy the generic requirement
            let x = T::from(rng.random_range(bounds_min[0].to_f64().unwrap()..(bounds_max[0].to_f64().unwrap()))).unwrap();
            let y = T::from(rng.random_range(bounds_min[1].to_f64().unwrap()..(bounds_max[1].to_f64().unwrap()))).unwrap();
            
            // 2. Keep the base of the mountain at the minimum Z (the floor)
            let z = bounds_min[2];
            let center = arr1(&[x, y, z]);

            // 3. Pick random dimensions
            let radius = T::from(rng.random_range(min_radius.to_f64().unwrap()..max_radius.to_f64().unwrap())).unwrap();
            let height = T::from(rng.random_range(min_height.to_f64().unwrap()..max_height.to_f64().unwrap())).unwrap();
            let sides = rng.random_range(min_sides..=max_sides);

            // 4. Create the mountain geometry
            let mountain_points = Self::mountain_points(center, radius, height, sides);
            
            // 5. Wrap in an Object
            let neutral = Neutral::new(attenuation);
            let mountain_object = Object::new(
                vec![mountain_points], 
                vec![TypeFace::Neutral(neutral)]
            );

            map.push(mountain_object);
        }
        map
    }
}

pub fn get_positions(
    max_x_i: u32,
    max_y_i: u32,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    z: f64,
    theta_transceiver: f64,
    phi_transceiver: f64,
    length_x_transceiver: f64,
    length_y_transceiver: f64,
    wavelength: f64,
    precision: (u32, u32),
) -> Vec<(ndarray::Array1<f64>, (f64, f64), Vec<ndarray::Array1<f64>>, Beamforming<f64>)> {
    let mut positions = Vec::new();
    for x_i in 0..max_x_i {
        for y_i in 0..max_y_i {
            let x = min_x + (max_x - min_x) * f64::from(x_i) / f64::from(max_x_i);
            let y = min_y + (max_y - min_y) * f64::from(y_i) / f64::from(max_y_i);

            // Matrices to apply a rotation around axis y, then around z
            let mat_rotation_y = arr2(&[
                [theta_transceiver.cos(), 0.0, theta_transceiver.sin()],
                [0.0, 1.0, 0.0],
                [-theta_transceiver.sin(), 0.0, theta_transceiver.cos()],
            ]);
            let mat_rotation_z = arr2(&[
                [phi_transceiver.cos(), -phi_transceiver.sin(), 0.0],
                [phi_transceiver.sin(), phi_transceiver.cos(), 0.0],
                [0.0, 0.0, 1.0],
            ]);

            let positions_3_points = arr2(&[
                [-length_x_transceiver / 2.0, -length_y_transceiver / 2.0, 0.0],
                [length_x_transceiver / 2.0, -length_y_transceiver / 2.0, 0.0],
                [-length_x_transceiver / 2.0, length_y_transceiver / 2.0, 0.0],
            ]);

            // Center of the transceiver
            let position_center = arr1(&[x, y, z]);

            let positions_3_points_rotated = positions_3_points.dot(&mat_rotation_y.t()).dot(&mat_rotation_z.t());

            let (vec, _) = positions_3_points_rotated.into_raw_vec_and_offset();

            let position_bottom_left = arr1(&vec[0..3]) + &position_center;
            let position_bottom_right = arr1(&vec[3..6]) + &position_center;
            let position_top_right = arr1(&vec[6..9]) + &position_center;

            let beamforming = Beamforming::new_rectangular_antenna(length_x_transceiver, length_y_transceiver, wavelength, precision);

            positions.push((
                position_center,
                (phi_transceiver, theta_transceiver),
                vec![position_bottom_left, position_bottom_right, position_top_right],
                beamforming,
            ));
        }
    }
    positions
}
