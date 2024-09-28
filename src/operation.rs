use crate::Tensor;

type ModelResult<T> = Result<<T as Tensor>::ModelOfTensor, String>;

#[derive(Clone, Copy, Debug)]
pub struct DiffableOperation<T: Tensor> {
    pub output_tensor: fn(&[T::ModelOfTensor]) -> ModelResult<T>,
    pub forward: fn(&[&T], &mut T),
    pub backprop: fn(output_grd: &T, inputs: &mut [&mut T]),
}
