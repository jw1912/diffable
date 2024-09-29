use std::cell::RefCell;

use crate::{Node, Tensor};

pub trait DiffableOperation<Tensor, Ctx, Model> {
    fn output_tensor(&self, inputs: &[Model]) -> Result<Model, String>;

    fn forward(&self, ctx: &mut Ctx, inputs: &[&Tensor], output: &mut Tensor);

    fn backward(&self, ctx: &mut Ctx, output_grad: &Tensor, inputs: &mut [&mut Tensor]);
}

#[derive(Debug)]
pub struct OperationPayload<T: Tensor> {
    operation: T::DiffableOperation,
    inputs: Vec<Node>,
    output: Node,
}

#[derive(Debug)]
pub struct OperationQueue<T: Tensor, const BACKPROP: bool> {
    queue: Vec<OperationPayload<T>>,
}

impl<T: Tensor, const BACKPROP: bool> OperationQueue<T, BACKPROP> {
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    pub fn push(&mut self, operation: T::DiffableOperation, inputs: &[Node], output: Node) {
        self.queue.push(OperationPayload {
            operation,
            inputs: inputs.to_vec(),
            output,
        });
    }
}

impl<T: Tensor> OperationQueue<T, false> {
    pub fn execute_on(&self, ctx: &mut T::ExecutionContext, graph: &mut [RefCell<T>]) {
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

            operation.forward(ctx, &inputs, &mut output);
        }
    }
}

impl<T: Tensor> OperationQueue<T, true> {
    pub fn execute_on(&self, ctx: &mut T::ExecutionContext, graph: &mut [RefCell<T>]) {
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

            operation.backward(ctx, &output, &mut inputs);
        }
    }
}
