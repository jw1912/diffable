use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Debug,
    ops::Index,
};

use crate::{
    operation::{BackwardFunc, ForwardFunc, OperationQueue},
    DiffableOperation, Graph, Tensor,
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Node(pub(crate) usize);

#[derive(Debug)]
pub struct NodeData<T: Tensor> {
    own: Node,
    id: Option<String>,
    vals: T::ModelOfTensor,
    requires_grad: bool,
    parent_operation: Option<DiffableOperation<T>>,
    parent_nodes: Vec<Node>,
}

impl<T: Tensor> NodeData<T> {
    pub fn new(
        id: Option<String>,
        operation: Option<DiffableOperation<T>>,
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
        operation: DiffableOperation<T>,
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

        let output_tensor = operation.output_tensor;
        match output_tensor(&input_vals) {
            Ok(vals) => self.create_node(NodeData::new(None, Some(operation), vals, true, inputs)),
            Err(s) => panic!("{s}"),
        }
    }

    fn set_of_leaves(&self) -> HashSet<Node> {
        self.inputs
            .union(&self.weights)
            .copied()
            .collect::<HashSet<Node>>()
    }

    fn set_of_edges(&self) -> HashSet<(Node, Node)> {
        self.nodes
            .iter()
            .map(|node_data| {
                node_data
                    .parent_nodes
                    .iter()
                    .map(|src| (*src, node_data.own))
                    .collect::<Vec<_>>()
            })
            .fold(Vec::new(), |mut x, y| {
                x.extend_from_slice(&y);
                x
            })
            .into_iter()
            .collect()
    }

    fn topological_sort_of_nodes(&self) -> Vec<Node> {
        let mut leaves = self.set_of_leaves();

        let mut sorted = Vec::new();

        let mut set_of_edges = self.set_of_edges();

        while !leaves.is_empty() {
            let n = *leaves.iter().next().unwrap();
            leaves.remove(&n);

            sorted.push(n);

            let edges_from_n = set_of_edges
                .iter()
                .copied()
                .filter(|(src, _)| *src == n)
                .collect::<Vec<_>>();

            for edge in edges_from_n {
                set_of_edges.remove(&edge);
                let m = edge.1;

                if !set_of_edges.iter().any(|(_, dest)| *dest == m) {
                    leaves.insert(m);
                }
            }
        }

        assert!(set_of_edges.is_empty(), "GraphBuilder contains a cycle!");

        sorted
    }

    fn build_forward(&self, sorted_nodes: &[Node]) -> OperationQueue<ForwardFunc<T>> {
        let mut queue = OperationQueue::new();

        for &node in sorted_nodes {
            let data = &self[node];

            if let Some(operation) = &data.parent_operation {
                queue.push(operation.forward, &data.parent_nodes, node);
            }
        }

        queue
    }

    fn build_backward(&self, sorted_nodes: &[Node]) -> OperationQueue<BackwardFunc<T>> {
        let mut queue = OperationQueue::new();

        for &node in sorted_nodes.iter().rev() {
            let data = &self[node];

            if let Some(operation) = &data.parent_operation {
                queue.push(operation.backprop, &data.parent_nodes, node);
            }
        }

        queue
    }

    pub fn build(&self) -> Graph<T> {
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

        let sorted = self.topological_sort_of_nodes();

        let forward = self.build_forward(&sorted);
        let backward = self.build_backward(&sorted);

        Graph {
            nodes,
            root,
            inputs,
            weights,
            forward,
            backward,
        }
    }
}
