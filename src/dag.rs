// ============================================================
// dag.rs — Dependency DAG: 위상 정렬 + TX 분할
// ============================================================

use std::collections::{HashMap, HashSet, VecDeque};
use crate::types::{TxId, DagNode, ShardId, NUM_SHARDS};

/// DAG 기반 의존 그래프
pub struct DependencyDag {
    /// tx_id → DagNode
    nodes: HashMap<TxId, DagNode>,
}

impl DependencyDag {
    pub fn new(nodes: Vec<DagNode>) -> Self {
        let map = nodes.into_iter().map(|n| (n.tx_id, n)).collect();
        Self { nodes: map }
    }

    /// Kahn's algorithm으로 위상 정렬된 레이어 반환
    /// 같은 레이어 안의 TX는 서로 독립 → 병렬 실행 가능
    pub fn topological_layers(&self) -> Vec<Vec<TxId>> {
        // in-degree 계산
        let mut in_degree: HashMap<TxId, usize> =
            self.nodes.keys().map(|&id| (id, 0)).collect();

        for node in self.nodes.values() {
            for &dep in &node.deps {
                // dep → node.tx_id 방향
                *in_degree.entry(node.tx_id).or_insert(0) += 0; // already inserted
                let _ = dep; // dep 노드는 이미 존재
            }
        }

        // 실제 in-degree 재계산 (node.deps = 이 노드가 의존하는 노드들)
        let mut in_deg: HashMap<TxId, usize> =
            self.nodes.keys().map(|&id| (id, 0)).collect();
        for node in self.nodes.values() {
            for &dep in &node.deps {
                // dep → node 엣지 → node의 in-degree 증가
                if in_deg.contains_key(&node.tx_id) {
                    *in_deg.get_mut(&node.tx_id).unwrap() += 1;
                }
                let _ = dep;
            }
        }

        let mut queue: VecDeque<TxId> = in_deg
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();

        // 역방향 에지 맵: dep → 이것에 의존하는 tx들
        let mut rev: HashMap<TxId, Vec<TxId>> = HashMap::new();
        for node in self.nodes.values() {
            for &dep in &node.deps {
                rev.entry(dep).or_default().push(node.tx_id);
            }
        }

        let mut layers: Vec<Vec<TxId>> = Vec::new();
        let mut visited = HashSet::new();

        while !queue.is_empty() {
            let layer_size = queue.len();
            let mut layer = Vec::with_capacity(layer_size);
            for _ in 0..layer_size {
                let tx = queue.pop_front().unwrap();
                layer.push(tx);
                visited.insert(tx);
                if let Some(dependents) = rev.get(&tx) {
                    for &dep_tx in dependents {
                        let d = in_deg.get_mut(&dep_tx).unwrap();
                        *d -= 1;
                        if *d == 0 {
                            queue.push_back(dep_tx);
                        }
                    }
                }
            }
            layers.push(layer);
        }

        layers
    }

    /// 각 TX를 샤드에 라운드-로빈 예비 배정 (optimizer가 이후 조정)
    pub fn initial_partition(&self, layers: &[Vec<TxId>]) -> HashMap<TxId, ShardId> {
        let mut assignment = HashMap::new();
        let mut counter = 0usize;
        for layer in layers {
            for &tx_id in layer {
                assignment.insert(tx_id, counter % NUM_SHARDS);
                counter += 1;
            }
        }
        assignment
    }
}
