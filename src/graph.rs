use std::{cell::RefCell, collections::HashMap, fmt::Display};

use crate::{
    operation::{BackwardFunc, ForwardFunc, OperationQueue},
    Node, Tensor,
};

pub struct Graph<T: Tensor> {
    pub(crate) nodes: Vec<RefCell<T>>,
    pub(crate) root: Node,
    pub(crate) inputs: HashMap<String, Node>,
    pub(crate) weights: HashMap<String, Node>,
    pub(crate) forward: OperationQueue<ForwardFunc<T>>,
    pub(crate) backward: OperationQueue<BackwardFunc<T>>,
}

impl<T: Tensor> Display for Graph<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.nodes)
    }
}

impl<T: Tensor> Graph<T> {
    pub fn forward(&mut self) -> f32 {
        self.forward.execute_on(&mut self.nodes);
        self.nodes[self.root.0].borrow().get_scalar().unwrap()
    }

    pub fn backward(&mut self) {
        self.nodes[self.root.0].get_mut().set_grad_to_unit();
        self.backward.execute_on(&mut self.nodes);
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
