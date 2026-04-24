//! Graph connectivity metric (Luecken 2022).
//!
//! For each unique label ℓ:
//!  1. Take the subgraph induced on cells with label ℓ, with edges from
//!     the provided k-NN graph (both endpoints must share the label).
//!  2. Find the size of the largest connected component (BFS).
//!  3. Compute `frac_ℓ = largest_cc_size / n_cells_with_label_ℓ`.
//!
//! Return the mean of `frac_ℓ` over all labels with ≥ 1 cell.
//!
//! Range: [0, 1]. 1 means every cell of each label is reachable from
//! every other cell of that label via the k-NN graph — i.e. the
//! integration preserves within-cell-type connectivity.

use std::collections::{BTreeMap, VecDeque};

use ndarray::{ArrayView1, ArrayView2};

/// Mean fraction-in-largest-CC across labels.
///
/// * `knn_indices` shape (n_cells, k), `u32::MAX` marks padding.
/// * `labels` shape (n_cells,) — labels (often cell-type codes).
///
/// Labels with zero cells are ignored (cannot happen if derived from the
/// same array) and labels with exactly one cell trivially contribute 1.
pub fn graph_connectivity(knn_indices: ArrayView2<u32>, labels: ArrayView1<i32>) -> f64 {
    let n_cells = knn_indices.nrows();
    assert_eq!(
        labels.len(),
        n_cells,
        "labels length must equal knn_indices rows"
    );

    let mut label_to_cells: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        label_to_cells.entry(l).or_default().push(i as u32);
    }

    // Build reverse adjacency: for each cell u, which cells have u as a
    // forward neighbor. scib-metrics and scipy both treat the k-NN graph
    // as undirected for connectivity, so BFS must follow edges in both
    // directions.
    let k = knn_indices.ncols();
    let mut rev_counts = vec![0usize; n_cells];
    for u in 0..n_cells {
        for j in 0..k {
            let v = knn_indices[[u, j]];
            if v == u32::MAX {
                continue;
            }
            let vi = v as usize;
            if vi < n_cells {
                rev_counts[vi] += 1;
            }
        }
    }
    // CSR-style layout for reverse neighbors
    let mut rev_offsets = vec![0usize; n_cells + 1];
    for i in 0..n_cells {
        rev_offsets[i + 1] = rev_offsets[i] + rev_counts[i];
    }
    let total_rev = rev_offsets[n_cells];
    let mut rev_data = vec![0u32; total_rev];
    let mut cursor = rev_offsets.clone();
    for u in 0..n_cells {
        for j in 0..k {
            let v = knn_indices[[u, j]];
            if v == u32::MAX {
                continue;
            }
            let vi = v as usize;
            if vi < n_cells {
                rev_data[cursor[vi]] = u as u32;
                cursor[vi] += 1;
            }
        }
    }

    let mut fracs: Vec<f64> = Vec::with_capacity(label_to_cells.len());
    for (_lbl, cells) in label_to_cells.iter() {
        if cells.is_empty() {
            continue;
        }
        if cells.len() == 1 {
            fracs.push(1.0);
            continue;
        }

        let mut in_label = vec![false; n_cells];
        for &c in cells {
            in_label[c as usize] = true;
        }
        let mut visited = vec![false; n_cells];
        let mut largest = 0usize;
        for &start in cells {
            if visited[start as usize] {
                continue;
            }
            let mut queue: VecDeque<u32> = VecDeque::new();
            queue.push_back(start);
            visited[start as usize] = true;
            let mut size = 0usize;
            while let Some(u) = queue.pop_front() {
                size += 1;
                let ui = u as usize;
                // Forward edges u → v
                for &v in knn_indices.row(ui).iter() {
                    if v == u32::MAX {
                        continue;
                    }
                    let vi = v as usize;
                    if vi < n_cells && in_label[vi] && !visited[vi] {
                        visited[vi] = true;
                        queue.push_back(v);
                    }
                }
                // Reverse edges v → u (treat graph as undirected)
                let rs = rev_offsets[ui];
                let re = rev_offsets[ui + 1];
                for &v in &rev_data[rs..re] {
                    let vi = v as usize;
                    if vi < n_cells && in_label[vi] && !visited[vi] {
                        visited[vi] = true;
                        queue.push_back(v);
                    }
                }
            }
            if size > largest {
                largest = size;
            }
        }
        fracs.push(largest as f64 / cells.len() as f64);
    }
    if fracs.is_empty() {
        return 0.0;
    }
    fracs.iter().sum::<f64>() / fracs.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn fully_connected_single_label_gives_one() {
        // 4 cells, all label 0, everybody points to the next → one
        // connected component spanning all cells.
        let knn: Array2<u32> = array![[1, 2], [2, 3], [0, 3], [0, 1]];
        let lbl: Array1<i32> = array![0, 0, 0, 0];
        let r = graph_connectivity(knn.view(), lbl.view());
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn two_disjoint_blocks_same_label() {
        // Cells 0,1,2 are one clique; cells 3,4,5 are another — no edges
        // across. Same label → CC sizes 3 each, largest = 3; 3/6 = 0.5.
        let knn: Array2<u32> = array![[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4],];
        let lbl: Array1<i32> = array![0, 0, 0, 0, 0, 0];
        let r = graph_connectivity(knn.view(), lbl.view());
        assert!((r - 0.5).abs() < 1e-12, "r = {}", r);
    }

    #[test]
    fn cross_label_edges_dont_count() {
        // Cells 0,1 label A; 2,3 label B. Edges only cross-label → each
        // label gives 2 CCs of size 1 each → largest_cc = 1 per label,
        // fraction = 1/2 each → mean = 0.5.
        let knn: Array2<u32> = array![[2, 3], [2, 3], [0, 1], [0, 1]];
        let lbl: Array1<i32> = array![0, 0, 1, 1];
        let r = graph_connectivity(knn.view(), lbl.view());
        assert!((r - 0.5).abs() < 1e-12, "r = {}", r);
    }

    #[test]
    fn padding_ignored() {
        let knn: Array2<u32> = array![[1, u32::MAX], [0, u32::MAX]];
        let lbl: Array1<i32> = array![0, 0];
        let r = graph_connectivity(knn.view(), lbl.view());
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn two_labels_one_connected_one_split() {
        // Label 0: 3 cells fully connected → frac = 1
        // Label 1: 2 cells no edges → frac = 1/2
        // mean = 0.75
        let knn: Array2<u32> = array![
            [1, 2],
            [0, 2],
            [0, 1],
            [u32::MAX, u32::MAX],
            [u32::MAX, u32::MAX]
        ];
        let lbl: Array1<i32> = array![0, 0, 0, 1, 1];
        let r = graph_connectivity(knn.view(), lbl.view());
        assert!((r - 0.75).abs() < 1e-12, "r = {}", r);
    }
}
