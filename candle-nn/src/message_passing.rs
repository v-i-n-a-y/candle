//! GNN message passing primitives built on the fused `gnn_scatter_add` kernel.

use candle::{DType, IndexOp, Result, Tensor};

/// Aggregate messages from source nodes to destination nodes (sum reduction).
///
/// For each destination node `v`, computes:
///   `out[v] = sum_{(u,v) in edges} node_features[u]`
///
/// # Arguments
/// * `node_features` -- `[N, D]` node feature matrix.
/// * `edge_index` -- `[2, E]` edge index tensor (row 0 = src, row 1 = dst), dtype u32/i32/i64.
/// * `n_nodes` -- total number of nodes `N` (needed because isolated nodes have no edges).
///
/// # Returns
/// Aggregated features `[N, D]` -- each node accumulates neighbour features.
pub fn aggregate_sum(
    node_features: &Tensor,
    edge_index: &Tensor,
    n_nodes: usize,
) -> Result<Tensor> {
    let src_idx = edge_index.i(0)?; // [E]
    let dst_idx = edge_index.i(1)?; // [E]
    let src_feat = node_features.index_select(&src_idx, 0)?; // [E, D]
    src_feat.gnn_scatter_add(&dst_idx, n_nodes) // [N, D]
}

/// Aggregate messages from source nodes to destination nodes (mean reduction).
///
/// For each destination node `v`, computes:
///   `out[v] = (1 / in_degree(v)) * sum_{(u,v) in edges} node_features[u]`
///
/// Nodes with degree zero are left at zero (no divide-by-zero).
///
/// # Arguments
/// * `node_features` -- `[N, D]` node feature matrix.
/// * `edge_index` -- `[2, E]` edge index tensor (row 0 = src, row 1 = dst), dtype u32/i32/i64.
/// * `n_nodes` -- total number of nodes `N`.
///
/// # Returns
/// Mean-aggregated features `[N, D]`.
pub fn aggregate_mean(
    node_features: &Tensor,
    edge_index: &Tensor,
    n_nodes: usize,
) -> Result<Tensor> {
    let sum = aggregate_sum(node_features, edge_index, n_nodes)?;

    // Compute in-degree for each destination node.
    let dst_idx = edge_index.i(1)?; // [E]
    // Build [E, 1] ones tensor and scatter-add to get [N, 1] in-degree counts.
    let e = dst_idx.dim(0)?;
    let ones = Tensor::ones((e, 1), DType::F32, node_features.device())?;
    let degree = ones.gnn_scatter_add(&dst_idx, n_nodes)?; // [N, 1]
    let degree = degree.clamp(1f32, f32::MAX)?; // avoid div-by-zero for isolated nodes
    let sum = sum.to_dtype(DType::F32)?;
    sum.broadcast_div(&degree)
}
