use diffable::{DiffableOperation, Graph, GraphBuilder, Node, Tensor};

fn main() {
    let mut graph = network();

    let lr = 0.01;

    graph.store_weights("a", &Float::from(1.0));
    graph.store_weights("b", &Float::from(0.0));

    let mut batch_loss = 1f32;

    while batch_loss > 0.01 {
        batch_loss = 0.0;

        for (input, target) in [(0.0, 3.0), (2.0, 7.0)] {
            graph.store_input("x", &Float::from(input));
            graph.store_input("y", &Float::from(target));

            let loss = graph.forward();

            batch_loss += loss;

            graph.zero_grads();
            graph.backward();

            for id in graph.weight_ids() {
                let weight = graph.get_weights_mut(&id);
                weight.val -= lr * weight.grad.unwrap();
            }
        }

        println!("Loss: {batch_loss}")
    }

    for id in graph.weight_ids() {
        println!("Weight '{id}': {:?}", graph.get_weights(&id).val);
    }
}

fn network() -> Graph<Float> {
    let mut builder = GraphBuilder::default();

    let a = builder.create_weights("a", ());
    let b = builder.create_weights("b", ());

    let x = builder.create_input("x", ());
    let y = builder.create_input("y", ());

    let c = Operation::mul(&mut builder, a, x);
    let pred = Operation::add(&mut builder, c, b);
    let diff = Operation::sub(&mut builder, pred, y);
    Operation::abs(&mut builder, diff);

    builder.build()
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Float {
    val: f32,
    grad: Option<f32>,
}

impl From<f32> for Float {
    fn from(val: f32) -> Self {
        Float { val, grad: None }
    }
}

impl From<Float> for () {
    fn from(_: Float) -> Self {}
}

impl Tensor for Float {
    type ModelOfTensor = ();

    fn new(_: Self::ModelOfTensor, requires_grad: bool) -> Self {
        Self {
            val: 0.0,
            grad: if requires_grad { Some(0.0) } else { None },
        }
    }

    fn get_scalar(&self) -> Option<f32> {
        Some(self.val)
    }

    fn copy_values_into(&self, dest: &mut Self) {
        dest.val = self.val;
    }

    fn zero_grad(&mut self) {
        if let Some(grad) = self.grad.as_mut() {
            *grad = 0.0;
        }
    }

    fn set_grad_to_unit(&mut self) {
        *self.grad.as_mut().unwrap() = 1.0;
    }
}

pub struct Operation;
impl Operation {
    pub fn add(graph: &mut GraphBuilder<Float>, a: Node, b: Node) -> Node {
        graph.create_result_of_operation(add(), &[a, b])
    }

    pub fn sub(graph: &mut GraphBuilder<Float>, a: Node, b: Node) -> Node {
        graph.create_result_of_operation(sub(), &[a, b])
    }

    pub fn mul(graph: &mut GraphBuilder<Float>, a: Node, b: Node) -> Node {
        graph.create_result_of_operation(mul(), &[a, b])
    }

    pub fn abs(graph: &mut GraphBuilder<Float>, a: Node) -> Node {
        graph.create_result_of_operation(abs(), &[a])
    }
}

fn add() -> DiffableOperation<Float> {
    DiffableOperation {
        output_tensor: add::is_valid,
        forward: add::forward,
        backprop: add::backprop,
    }
}

mod add {
    use super::*;

    pub fn is_valid(inputs: &[()]) -> Result<(), String> {
        if inputs.len() != 2 {
            Err(String::from("Invalid number of arguments!"))
        } else {
            Ok(())
        }
    }

    pub fn forward(inputs: &[&Float], output: &mut Float) {
        output.val = inputs[0].val + inputs[1].val;
    }

    pub fn backprop(output: &Float, inputs: &mut [&mut Float]) {
        let output_grad = output.grad.unwrap();

        for input in inputs {
            if let Some(grd) = input.grad.as_mut() {
                *grd += output_grad;
            }
        }
    }
}

fn sub() -> DiffableOperation<Float> {
    DiffableOperation {
        output_tensor: sub::is_valid,
        forward: sub::forward,
        backprop: sub::backprop,
    }
}

mod sub {
    use super::*;

    pub fn is_valid(inputs: &[()]) -> Result<(), String> {
        if inputs.len() != 2 {
            Err(String::from("Invalid number of arguments!"))
        } else {
            Ok(())
        }
    }

    pub fn forward(inputs: &[&Float], output: &mut Float) {
        output.val = inputs[0].val - inputs[1].val;
    }

    pub fn backprop(output: &Float, inputs: &mut [&mut Float]) {
        let output_grad = output.grad.unwrap();

        for (input, sgn) in inputs.iter_mut().zip([1.0, -1.0].iter()) {
            if let Some(grd) = input.grad.as_mut() {
                *grd += sgn * output_grad;
            }
        }
    }
}

fn mul() -> DiffableOperation<Float> {
    DiffableOperation {
        output_tensor: mul::is_valid,
        forward: mul::forward,
        backprop: mul::backprop,
    }
}

mod mul {
    use super::*;

    pub fn is_valid(inputs: &[()]) -> Result<(), String> {
        if inputs.len() != 2 {
            Err(String::from("Invalid number of arguments!"))
        } else {
            Ok(())
        }
    }

    pub fn forward(inputs: &[&Float], output: &mut Float) {
        output.val = inputs[0].val * inputs[1].val;
    }

    pub fn backprop(output: &Float, inputs: &mut [&mut Float]) {
        let output_grad = output.grad.unwrap();

        for (a, b) in [(0, 1), (1, 0)] {
            if let Some(grd) = inputs[a].grad.as_mut() {
                *grd += output_grad * inputs[b].val;
            }
        }
    }
}

fn abs() -> DiffableOperation<Float> {
    DiffableOperation {
        output_tensor: abs::is_valid,
        forward: abs::forward,
        backprop: abs::backprop,
    }
}

mod abs {
    use super::*;

    pub fn is_valid(inputs: &[()]) -> Result<(), String> {
        if inputs.len() != 1 {
            Err(String::from("Invalid number of arguments!"))
        } else {
            Ok(())
        }
    }

    pub fn forward(inputs: &[&Float], output: &mut Float) {
        output.val = inputs[0].val.abs();
    }

    pub fn backprop(output: &Float, inputs: &mut [&mut Float]) {
        let output_grad = output.grad.unwrap();

        if let Some(grd) = inputs[0].grad.as_mut() {
            *grd += output_grad
                * if inputs[0].val > 0.0 {
                    1.0
                } else if inputs[0].val == 0.0 {
                    0.0
                } else {
                    -1.0
                };
        }
    }
}
