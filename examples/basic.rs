use diffable::{DiffableOperation, Graph, GraphBuilder, Node, Tensor};

fn main() {
    let mut graph = network();

    graph.store_input("a", &Float::from(3.0));
    graph.store_input("b", &Float::from(2.0));

    graph.store_weights("w", &Float::from(5.0));

    let out = graph.forward();

    println!("{out}");
}

fn network() -> Graph<Float> {
    let mut builder = GraphBuilder::default();

    let a = builder.create_input("a", ());
    let b = builder.create_input("b", ());

    let w = builder.create_weights("w", ());

    let c = Operation::mul(&mut builder, a, b);
    let d = Operation::mul(&mut builder, w, c);
    let e = Operation::add(&mut builder, d, b);
    Operation::add(&mut builder, e, a);

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
}

pub struct Operation;
impl Operation {
    pub fn add(graph: &mut GraphBuilder<Float>, a: Node, b: Node) -> Node {
        graph.create_result_of_operation(add(), &[a, b])
    }

    pub fn mul(graph: &mut GraphBuilder<Float>, a: Node, b: Node) -> Node {
        graph.create_result_of_operation(mul(), &[a, b])
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

    pub fn backprop(output_grd: &Float, inputs: &mut [&mut Float]) {
        for input in inputs {
            if let Some(grd) = input.grad.as_mut() {
                *grd += output_grd.val;
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

    pub fn backprop(_: &Float, _: &mut [&mut Float]) {}
}
