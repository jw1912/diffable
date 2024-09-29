use std::fmt::Debug;

use crate::DiffableOperation;

pub trait Tensor: Debug + Default {
    type ModelOfTensor: Copy + Debug + From<Self>;
    type ExecutionContext: Debug;
    type DiffableOperation: Copy
        + Debug
        + DiffableOperation<Self, Self::ExecutionContext, Self::ModelOfTensor>;

    fn new(desc: Self::ModelOfTensor, requires_grad: bool) -> Self;

    fn get_scalar(&self) -> Option<f32>;

    fn copy_values_into(&self, dest: &mut Self);

    fn zero_grad(&mut self);

    fn set_grad_to_unit(&mut self);
}
