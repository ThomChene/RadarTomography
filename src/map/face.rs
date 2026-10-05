use ndarray::{arr1, Array1, NdFloat};
use std::fmt;

use crate::algorithms::signal;
use super::object::GroupPoints;
use super::ray::Ray;

/// Three points different from one another constitute a triangular `Face` that belongs in a 2D plane.
/// We compute the normal of the plane (it will be oriented such that points p1,p2,p3 are in anti clockwise order).
/// The normal (a,b,c) and the offset (d) fully determine the plane (ax + by + cz + d = 0), but we still need the points to check if a ray intersects the plane within the boundaries of the triangle.
/// 
pub struct Face<T:NdFloat>{
    normal: Array1<T>,
    offset: T,
    p1: Array1<T>,
    p2: Array1<T>,
    p3: Array1<T>,
    squared_norm_first_vector: T,
    squared_norm_second_vector: T,
    scalar_product_vectors: T,
    type_face: TypeFace<T>,
}

/// Display:
/// 
impl<'a,T:NdFloat> fmt::Display for Face<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Face constituted of plane with:\n")?;
        write!(f, "normal: {}\n", &self.normal)?;
        write!(f, "offset: {}\n", &self.offset)
    }
}

/// New and getter:
/// 
impl<'a ,T:NdFloat> Face<T>{
    pub fn new(points: (Array1<T>, Array1<T>, Array1<T>),type_face: TypeFace<T>)->Self{
        // Cross Product to compute the normal, then normalize it
        let (p1,p2,p3) = points;
        let vector1 = &p2-&p1;
        let vector2 = &p3-&p1;

        if vector1==vector2 {
            panic!("Points should be different from one another in a Face");
        }

        let value1 = vector1[1]*vector2[2]-vector1[2]*vector2[1];
        let value2 = vector1[2]*vector2[0]-vector1[0]*vector2[2];
        let value3 = vector1[0]*vector2[1]-vector1[1]*vector2[0];
        let normalization_factor = T::sqrt(value1*value1 + value2*value2 + value3*value3);
        let normal = arr1(&[value1/normalization_factor,value2/normalization_factor,value3/normalization_factor]);
        let offset = -normal.dot(&p1);

        let squared_norm_first_vector = signal::scalar_prod(&vector1,&vector1);
        let squared_norm_second_vector = signal::scalar_prod(&vector2,&vector2);
        let scalar_product_vectors = signal::scalar_prod(&vector1,&vector2);

        Face{
            normal,
            offset,
            p1,
            p2,
            p3,
            squared_norm_first_vector,
            squared_norm_second_vector,
            scalar_product_vectors,
            type_face,
        }
    }

    pub fn get_values(&self)->(&Array1<T>,T,&Array1<T>,&Array1<T>,&Array1<T>,T,T,T){
        (&self.normal, self.offset, &self.p1, &self.p2, &self.p3, self.squared_norm_first_vector, self.squared_norm_second_vector, self.scalar_product_vectors)
    }

    pub fn ray_received(&self, ray: Ray<T>, closest_dist: T, intersection_face_ray: Array1<T>)->(Vec<Ray<T>>,Option<(T,T,usize)>){
        let (amplitude, distance, _, direction, min_amplitude_ray) = ray.get_values();
        let new_amplitude;
        let option;
        match &self.type_face{
            TypeFace::Neutral(neut)=> {
                new_amplitude = neut.attenuation/(closest_dist*closest_dist);
                option = None;},
            TypeFace::Receiver(rec)=> {
                new_amplitude = T::zero();
                option = Some((amplitude, distance, rec.index_receiver));},
        }
        let new_direction = direction*(-T::one());

        // if the amplitude of the ray is too low, does not create it
        if new_amplitude*amplitude > min_amplitude_ray{
            let out_ray= Ray::new(new_amplitude*amplitude, closest_dist + distance, intersection_face_ray, new_direction, min_amplitude_ray);
            (vec!(out_ray), option)
        }
        else{
            (Vec::new(), option)
        }
    }
}

/// The face can either attenuate the signal and scatter it or retro propagate it, or be a receiver and absorbs the signal received.
#[derive(Clone,Copy)]
pub enum TypeFace<T:NdFloat>{
    Neutral(Neutral<T>),
    Receiver(Receiver),
}
/// When multiple receiver exists, each of them has an index
#[derive(Clone,Copy)]
pub struct Receiver{
    index_receiver: usize,
}
#[derive(Clone,Copy)]
pub struct Neutral<T:NdFloat>{
    attenuation: T,
}

impl Receiver{
    pub fn new(index_receiver: usize)->Self{
        Receiver{
            index_receiver,
        }
    }
}

impl<T:NdFloat> Neutral<T>{
    pub fn new(attenuation: T)->Self{
        Neutral{
            attenuation,
        }
    }
}

/// Faces belonging to the same object and of the same type are stored in a vector and we add some functions to build those faces and update them if the object moves.
pub struct GroupFaces<T:NdFloat>{
    faces: Vec<Face<T>>,
}

impl<'a,T:NdFloat> GroupFaces<T>{
    pub fn get_faces(&self)->Vec<&Face<T>>{
        let mut vec = Vec::new();
        for face in self.faces.iter(){
            vec.push(face);
        }
        vec
    }
    pub fn new(group_points: &'a GroupPoints<T>, type_face: TypeFace<T>)->Self{
        let mut faces = Vec::new();
        for n_point in 0..group_points.get_number_faces(){
            let indices = group_points.get_indices(n_point);
            let points_cloned = (group_points.get_point(indices.0).clone(),group_points.get_point(indices.1).clone(),group_points.get_point(indices.2).clone());
            let face = Face::new(points_cloned,type_face);
            faces.push(face);
        }
        GroupFaces{
            faces,
        }
    }
}