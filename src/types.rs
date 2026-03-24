use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Transaction {
    pub id: u64,
    pub read_set: HashSet<String>,
    pub write_set: HashSet<String>,
}

pub struct DagNode {
    pub tx_id: u64,
    pub dependencies: Vec<u64>,
}

#[repr(align(64))]
pub struct PaddedState {
    pub value: u64,
}
