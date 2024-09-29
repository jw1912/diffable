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

    let c = mul(&mut builder, a, x);
    let pred = add(&mut builder, c, b);
    let offset = inc(&mut builder, pred, 1.0);
    let diff = sub(&mut builder, offset, y);
    let _abs = abs(&mut builder, diff);

    builder.build(())
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
    type ExecutionContext = ();
    type DiffableOperation = Operation;

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

fn add(graph: &mut GraphBuilder<Float>, a: Node, b: Node) -> Node {
    graph.create_result_of_operation(Operation::Add, &[a, b])
}

fn sub(graph: &mut GraphBuilder<Float>, a: Node, b: Node) -> Node {
    graph.create_result_of_operation(Operation::Sub, &[a, b])
}

fn mul(graph: &mut GraphBuilder<Float>, a: Node, b: Node) -> Node {
    graph.create_result_of_operation(Operation::Mul, &[a, b])
}

fn abs(graph: &mut GraphBuilder<Float>, a: Node) -> Node {
    graph.create_result_of_operation(Operation::Abs, &[a])
}

fn inc(graph: &mut GraphBuilder<Float>, a: Node, val: f32) -> Node {
    graph.create_result_of_operation(Operation::Inc(val), &[a])
}

#[derive(Clone, Copy, Debug)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Abs,
    Inc(f32),
}

impl DiffableOperation<Float, (), ()> for Operation {
    fn output_tensor(&self, inputs: &[()]) -> Result<(), String> {
        match self {
            Self::Add => add::is_valid(inputs),
            Self::Sub => sub::is_valid(inputs),
            Self::Mul => mul::is_valid(inputs),
            Self::Abs => abs::is_valid(inputs),
            Self::Inc(_) => inc::is_valid(inputs),
        }
    }

    fn forward(&self, _: &mut (), inputs: &[&Float], output: &mut Float) {
        match self {
            Self::Add => add::forward(inputs, output),
            Self::Sub => sub::forward(inputs, output),
            Self::Mul => mul::forward(inputs, output),
            Self::Abs => abs::forward(inputs, output),
            Self::Inc(val) => inc::forward(*val, inputs, output),
        }
    }

    fn backward(&self, _: &mut (), output_grad: &Float, inputs: &mut [&mut Float]) {
        match self {
            Self::Add => add::backprop(output_grad, inputs),
            Self::Sub => sub::backprop(output_grad, inputs),
            Self::Mul => mul::backprop(output_grad, inputs),
            Self::Abs => abs::backprop(output_grad, inputs),
            Self::Inc(_) => inc::backprop(output_grad, inputs),
        }
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

mod inc {
    use super::*;

    pub fn is_valid(inputs: &[()]) -> Result<(), String> {
        if inputs.len() != 1 {
            Err(String::from("Invalid number of arguments!"))
        } else {
            Ok(())
        }
    }

    pub fn forward(val: f32, inputs: &[&Float], output: &mut Float) {
        output.val = inputs[0].val.abs() + val;
    }

    pub fn backprop(output: &Float, inputs: &mut [&mut Float]) {
        let output_grad = output.grad.unwrap();

        if let Some(grd) = inputs[0].grad.as_mut() {
            *grd += output_grad;
        }
    }
}
