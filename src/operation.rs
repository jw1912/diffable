use std::cell::RefCell;

use crate::{Node, Tensor};

type ModelResult<T> = Result<<T as Tensor>::ModelOfTensor, String>;

#[derive(Clone, Copy, Debug)]
pub struct DiffableOperation<T: Tensor> {
    pub output_tensor: fn(&[T::ModelOfTensor]) -> ModelResult<T>,
    pub forward: ForwardFunc<T>,
    pub backprop: BackwardFunc<T>,
}

#[derive(Debug)]
pub struct OperationPayload<F> {
    operation: F,
    inputs: Vec<Node>,
    output: Node,
}

#[derive(Debug)]
pub struct OperationQueue<F> {
    queue: Vec<OperationPayload<F>>,
}

impl<F> OperationQueue<F> {
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    pub fn push(&mut self, operation: F, inputs: &[Node], output: Node) {
        self.queue.push(OperationPayload {
            operation,
            inputs: inputs.to_vec(),
            output,
        });
    }
}

pub type ForwardFunc<T> = fn(&<T as Tensor>::ExecutionContext, &[&T], &mut T);

impl<T: Tensor> OperationQueue<ForwardFunc<T>> {
    pub fn execute_on(&self, ctx: &T::ExecutionContext, graph: &mut [RefCell<T>]) {
        for OperationPayload {
            operation,
            inputs,
            output,
        } in &self.queue
        {
            let inputs = inputs
                .iter()
                .map(|node| graph[node.0].borrow())
                .collect::<Vec<_>>();

            let inputs = inputs
                .iter()
                .map(|ref_cell| &**ref_cell)
                .collect::<Vec<_>>();

            let mut output = graph[output.0].borrow_mut();

            operation(ctx, &inputs, &mut output);
        }
    }
}

pub type BackwardFunc<T> = fn(&<T as Tensor>::ExecutionContext, &T, &mut [&mut T]);

impl<T: Tensor> OperationQueue<BackwardFunc<T>> {
    pub fn execute_on(&self, ctx: &T::ExecutionContext, graph: &mut [RefCell<T>]) {
        for OperationPayload {
            operation,
            inputs,
            output,
        } in &self.queue
        {
            let mut inputs = inputs
                .iter()
                .map(|node| graph[node.0].borrow_mut())
                .collect::<Vec<_>>();

            let mut inputs = inputs
                .iter_mut()
                .map(|ref_cell| &mut **ref_cell)
                .collect::<Vec<_>>();

            let output = graph[output.0].borrow();

            operation(ctx, &output, &mut inputs);
        }
    }
}
