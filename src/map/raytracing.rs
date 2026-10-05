use ndarray::NdFloat;
use super::object::{Object, Transmitter};
use super::object;
use super::ray::Ray;

pub fn raytracing<T>(n_rays: u32, n_receiver: u32, transmitter: &Transmitter<T>, objects: &Vec<Object<T>>, min_amplitude_ray: T)->(Vec<Vec<T>>, Vec<Vec<T>>, Vec<Vec<(T,T,T)>>)
where
T:NdFloat,
{
    let mut rays: Vec<Ray<T>> = transmitter.create_rays(n_rays, min_amplitude_ray);

    let mut all_amplitudes_receivers = Vec::new();
    let mut all_distances_receivers = Vec::new();
    let mut to_plot_rays = Vec::new();
    for _receiver in 0..n_receiver{
        all_amplitudes_receivers.push(Vec::new());
        all_distances_receivers.push(Vec::new());
    }
    let faces = object::from_objects_to_faces(objects);
    while !rays.is_empty(){
        // Should be parallelized
        let ray  = rays.pop().unwrap();
        // faces = face_tree(ray,faces);

        // To plot rays
        let (_,_,origin,_,_) = ray.get_values();
        let begin_ray = (origin[0],origin[1],origin[2]);

        // Propagate the ray
        let (rays_reflected, option_received) = ray.propagate(&faces);

        if !rays_reflected.is_empty(){
            for new_ray in rays_reflected{
                // To plot rays
                let (_,_,end,_,_) = (new_ray).get_values();
                let end_ray = (end[0],end[1],end[2]);
                let vec = vec![begin_ray,end_ray];
                to_plot_rays.push(vec);

                // To return new rays
                rays.push(new_ray);
            }
        }

        match option_received{
            Some((amplitude,distance,index_receiver)) =>{
                all_amplitudes_receivers[index_receiver].push(amplitude);
                all_distances_receivers[index_receiver].push(distance);}
            None => (),}
    }
    (all_amplitudes_receivers,all_distances_receivers,to_plot_rays)
}