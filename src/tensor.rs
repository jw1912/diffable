use std::fmt::Debug;

pub trait Tensor: Debug + Default {
    type ModelOfTensor: Copy + Debug;

    fn new(desc: Self::ModelOfTensor, requires_grad: bool) -> Self;

    fn get_scalar(&self) -> Option<f32>;

    fn copy_values_into(&self, dest: &mut Self);

    fn zero_grad(&mut self);
}
