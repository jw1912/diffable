use std::{cell::RefCell, collections::HashMap, fmt::Debug};

use crate::{Node, Tensor};

pub struct Graph<T: Tensor> {
    pub(crate) nodes: Vec<RefCell<T>>,

    pub(crate) root: Node,
    pub(crate) inputs: HashMap<String, Node>,
    pub(crate) weights: HashMap<String, Node>,

    pub(crate) forward: OperationQueue<T>,
}

impl<T: Tensor> Graph<T> {
    pub fn forward(&mut self) -> f32 {
        self.forward.execute_on(&mut self.nodes);
        self.nodes[self.root.0].borrow().get_scalar().unwrap()
    }

    fn store_values(&mut self, node: Node, data: &T) {
        data.copy_values_into(self.nodes[node.0].get_mut());
    }

    pub fn store_input(&mut self, input: &str, data: &T) {
        self.store_values(self.inputs[input], data);
    }

    pub fn store_weights(&mut self, weights: &str, data: &T) {
        self.store_values(self.weights[weights], data);
    }
}

type Func<T> = fn(&[&T], &mut T);

#[derive(Debug)]
pub struct OperationPayload<T: Tensor> {
    operation: Func<T>,
    inputs: Vec<Node>,
    output: Node,
}

#[derive(Debug, Default)]
pub struct OperationQueue<T: Tensor> {
    queue: Vec<OperationPayload<T>>,
}

impl<T: Tensor> OperationQueue<T> {
    pub fn push(&mut self, operation: Func<T>, inputs: &[Node], output: Node) {
        self.queue.push(OperationPayload {
            operation,
            inputs: inputs.to_vec(),
            output,
        });
    }

    pub fn execute_on(&self, graph: &mut [RefCell<T>]) {
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

            operation(&inputs, &mut output);
        }
    }
}
