use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Debug,
    ops::Index,
};

use crate::{operation::OperationQueue, DiffableOperation, Graph, Tensor};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Node(pub(crate) usize);

#[derive(Debug)]
pub struct NodeData<T: Tensor> {
    own: Node,
    id: Option<String>,
    vals: T::ModelOfTensor,
    requires_grad: bool,
    parent_operation: Option<T::DiffableOperation>,
    parent_nodes: Vec<Node>,
}

impl<T: Tensor> NodeData<T> {
    pub fn new(
        id: Option<String>,
        operation: Option<T::DiffableOperation>,
        vals: T::ModelOfTensor,
        requires_grad: bool,
        parents: &[Node],
    ) -> Self {
        Self {
            id,
            own: Node(usize::MAX),
            vals,
            requires_grad,
            parent_operation: operation,
            parent_nodes: parents.to_vec(),
        }
    }
}

#[derive(Debug, Default)]
pub struct GraphBuilder<T: Tensor> {
    nodes: Vec<NodeData<T>>,
    roots: HashSet<Node>,
    inputs: HashSet<Node>,
    weights: HashSet<Node>,
    ids: HashSet<String>,
}

impl<T: Tensor> Index<Node> for GraphBuilder<T> {
    type Output = NodeData<T>;

    fn index(&self, index: Node) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl<T: Tensor> GraphBuilder<T> {
    pub fn create_node(&mut self, mut data: NodeData<T>) -> Node
    where
        T: Sized,
    {
        if let Some(id) = data.id.as_ref() {
            assert!(self.ids.insert(id.to_string()))
        }

        let node = Node(self.nodes.len());
        data.own = node;

        for parent in &data.parent_nodes {
            self.roots.remove(parent);
        }

        self.nodes.push(data);
        self.roots.insert(node);

        node
    }

    pub fn create_input(&mut self, id: &str, vals: T::ModelOfTensor) -> Node {
        let node = self.create_node(NodeData::new(Some(id.to_string()), None, vals, false, &[]));

        self.inputs.insert(node);

        node
    }

    pub fn create_weights(&mut self, id: &str, vals: T::ModelOfTensor) -> Node {
        let node = self.create_node(NodeData::new(Some(id.to_string()), None, vals, true, &[]));

        self.weights.insert(node);

        node
    }

    pub fn create_result_of_operation(
        &mut self,
        operation: T::DiffableOperation,
        inputs: &[Node],
    ) -> Node {
        let mut set = HashSet::new();
        assert!(
            inputs.iter().all(|node| set.insert(node)),
            "An operation will alias nodes on backprop!"
        );

        let input_vals = inputs
            .iter()
            .map(|node| self[*node].vals)
            .collect::<Vec<_>>();

        match operation.output_tensor(&input_vals) {
            Ok(vals) => self.create_node(NodeData::new(None, Some(operation), vals, true, inputs)),
            Err(s) => panic!("{s}"),
        }
    }

    fn build_forward(&self, nodes: &[Node]) -> OperationQueue<T, false> {
        let mut queue = OperationQueue::new();

        for &node in nodes {
            let data = &self[node];

            if let Some(operation) = data.parent_operation {
                queue.push(operation, &data.parent_nodes, node);
            }
        }

        queue
    }

    fn build_backward(&self, nodes: &[Node]) -> OperationQueue<T, true> {
        let mut queue = OperationQueue::new();

        for &node in nodes.iter().rev() {
            let data = &self[node];

            if let Some(operation) = data.parent_operation {
                queue.push(operation, &data.parent_nodes, node);
            }
        }

        queue
    }

    pub fn build(&self, execution_context: T::ExecutionContext) -> Graph<T> {
        assert_eq!(self.roots.len(), 1, "Graph must have a single output!");

        let root = *self.roots.iter().next().unwrap();
        assert!(self[root].requires_grad, "Output cannot be an input!");
        assert!(
            !self.weights.contains(&root),
            "Can't output trainable weights!"
        );

        let nodes = self
            .nodes
            .iter()
            .map(|node_data| RefCell::new(T::new(node_data.vals, node_data.requires_grad)))
            .collect::<Vec<_>>();

        let inputs = self
            .inputs
            .iter()
            .map(|&node| (self[node].id.clone().unwrap(), node))
            .collect::<HashMap<_, _>>();

        let weights = self
            .weights
            .iter()
            .map(|&node| (self[node].id.clone().unwrap(), node))
            .collect::<HashMap<_, _>>();

        let node_ids = self
            .nodes
            .iter()
            .map(|node_data| node_data.own)
            .collect::<Vec<_>>();

        let forward = self.build_forward(&node_ids);
        let backward = self.build_backward(&node_ids);

        Graph {
            nodes,
            root,
            inputs,
            weights,
            forward,
            backward,
            execution_context,
        }
    }
}
