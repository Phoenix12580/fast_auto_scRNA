//! Batch-balanced KNN — per-batch nearest neighbors over a PCA embedding.
//!
//! Given an `(n_cells, n_dims)` PCA matrix and per-cell batch labels, find
//! the `k_per_batch` nearest neighbors **within each batch** for every
//! cell. The concatenated `(indices, distances)` arrays feed downstream
//! UMAP connectivity construction (handled in the Python layer via
//! umap-learn's `fuzzy_simplicial_set`, which is where `bbknn` diverges
//! from vanilla scanpy neighbors).
//!
//! This MVP uses brute-force Euclidean distance. It is exact — top-k is
//! deterministic given `pca` and `batch_labels`. For scale (>50k cells)
//! the hot loop will be swapped to `hnsw-rs` in a later milestone; the
//! public `BbknnResult` layout stays stable.

use std::collections::BTreeMap;

use hnsw_rs::prelude::*;
use ndarray::{ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Result of a BBKNN run. Row-major contiguous buffers; the PyO3 layer
/// reshapes to `(n_cells, k_total)`.
pub struct BbknnResult {
    /// Global cell indices of neighbors, shape `(n_cells, k_total)` flat.
    /// Columns are ordered by batch (first `k_per_batch` cols are the
    /// smallest batch label, next are the second batch, etc.). Slots for
    /// batches smaller than `k_per_batch` are padded with `u32::MAX`.
    pub indices: Vec<u32>,
    /// Euclidean distances aligned with `indices`. Padded slots hold
    /// `f32::INFINITY`.
    pub distances: Vec<f32>,
    /// Sorted unique batch values; determines column order in output.
    pub batches: Vec<i32>,
    /// Requested `k_per_batch`; output has `batches.len() * k_per_batch`
    /// columns.
    pub k_per_batch: usize,
    /// Convenience: `batches.len() * k_per_batch`.
    pub k_total: usize,
}

#[inline]
fn squared_l2(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let mut s = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        s += d * d;
    }
    s
}

/// Exact brute-force batch-balanced KNN over a PCA embedding.
///
/// # Panics
///
/// * `batch_labels.len() != pca.nrows()`
/// * `k_per_batch == 0`
pub fn bbknn_brute(
    pca: ArrayView2<f32>,
    batch_labels: ArrayView1<i32>,
    k_per_batch: usize,
) -> BbknnResult {
    let n_cells = pca.nrows();
    assert_eq!(
        batch_labels.len(),
        n_cells,
        "batch_labels length must equal pca rows"
    );
    assert!(k_per_batch > 0, "k_per_batch must be > 0");

    // BTreeMap keeps batches in sorted ascending order so column layout is
    // fully determined by input labels (no HashMap hash-seed nondeterminism).
    let mut batch_to_cells: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
    for (i, &b) in batch_labels.iter().enumerate() {
        batch_to_cells.entry(b).or_default().push(i as u32);
    }
    let batches: Vec<i32> = batch_to_cells.keys().copied().collect();
    let n_batches = batches.len();
    let k_total = k_per_batch * n_batches;

    // Per-batch cell index slices in batch order.
    let per_batch_cells: Vec<&[u32]> = batches
        .iter()
        .map(|b| batch_to_cells[b].as_slice())
        .collect();

    let mut indices = vec![u32::MAX; n_cells * k_total];
    let mut distances = vec![f32::INFINITY; n_cells * k_total];

    indices
        .par_chunks_mut(k_total)
        .zip(distances.par_chunks_mut(k_total))
        .enumerate()
        .for_each(|(q, (idx_row, dist_row))| {
            let query = pca.row(q);
            let mut scratch: Vec<(f32, u32)> = Vec::new();
            for (bi, cells) in per_batch_cells.iter().enumerate() {
                if cells.is_empty() {
                    continue;
                }
                scratch.clear();
                scratch.reserve(cells.len());
                for &c in cells.iter() {
                    let d = squared_l2(query, pca.row(c as usize));
                    scratch.push((d, c));
                }
                let k_eff = k_per_batch.min(cells.len());
                if k_eff < scratch.len() {
                    scratch.select_nth_unstable_by(k_eff - 1, |a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                let top = &mut scratch[..k_eff];
                top.sort_unstable_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                let start = bi * k_per_batch;
                for (i, (d_sq, c)) in top.iter().enumerate() {
                    dist_row[start + i] = d_sq.sqrt();
                    idx_row[start + i] = *c;
                }
            }
        });

    BbknnResult {
        indices,
        distances,
        batches,
        k_per_batch,
        k_total,
    }
}

/// Approximate batch-balanced KNN via HNSW (one index per batch).
///
/// Much faster than `bbknn_brute` for batches ≳ 10k cells; at small N the
/// build overhead (M=16, ef_construction=200) makes brute force the
/// winner. Auto-selection lives in the PyO3 layer.
///
/// `ef_search` controls recall: must be ≥ `k_per_batch`; typical 4× k
/// (i.e. ≥ 12 for default k=3) to keep recall > 0.95.
///
/// Distances are Euclidean (matches `bbknn_brute`); `hnsw_rs` DistL2
/// returns the actual L2 value, not squared.
///
/// # Panics
///
/// * `batch_labels.len() != pca.nrows()`
/// * `k_per_batch == 0`
/// * `ef_search < k_per_batch`
pub fn bbknn_hnsw(
    pca: ArrayView2<f32>,
    batch_labels: ArrayView1<i32>,
    k_per_batch: usize,
    ef_search: usize,
) -> BbknnResult {
    let n_cells = pca.nrows();
    assert_eq!(
        batch_labels.len(),
        n_cells,
        "batch_labels length must equal pca rows"
    );
    assert!(k_per_batch > 0, "k_per_batch must be > 0");
    assert!(ef_search >= k_per_batch, "ef_search must be >= k_per_batch");

    let mut batch_to_cells: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
    for (i, &b) in batch_labels.iter().enumerate() {
        batch_to_cells.entry(b).or_default().push(i as u32);
    }
    let batches: Vec<i32> = batch_to_cells.keys().copied().collect();
    let n_batches = batches.len();
    let k_total = k_per_batch * n_batches;

    // Pre-materialize each cell's PCA row as Vec<f32> once — hnsw_rs insert
    // and search APIs want &Vec<f32>, and this also avoids re-allocating
    // per query inside rayon.
    let all_rows: Vec<Vec<f32>> = (0..n_cells).map(|i| pca.row(i).to_vec()).collect();

    let mut indices = vec![u32::MAX; n_cells * k_total];
    let mut distances = vec![f32::INFINITY; n_cells * k_total];

    for (bi, bval) in batches.iter().enumerate() {
        let cells = &batch_to_cells[bval];
        if cells.is_empty() {
            continue;
        }
        let k_eff = k_per_batch.min(cells.len());
        let ef = ef_search.max(k_eff);

        // Build HNSW for this batch. max_nb_connection=16, max_layer=16,
        // ef_construction=200 are the hnsw_rs / paper-recommended defaults.
        let hnsw = Hnsw::<f32, DistL2>::new(16, cells.len(), 16, 200, DistL2 {});
        let refs: Vec<(&Vec<f32>, usize)> = cells
            .iter()
            .enumerate()
            .map(|(within_idx, &global_idx)| (&all_rows[global_idx as usize], within_idx))
            .collect();
        hnsw.parallel_insert(&refs);

        // Query every cell against this batch's index.
        indices
            .par_chunks_mut(k_total)
            .zip(distances.par_chunks_mut(k_total))
            .enumerate()
            .for_each(|(q, (idx_row, dist_row))| {
                let nbrs = hnsw.search(&all_rows[q], k_eff, ef);
                let start = bi * k_per_batch;
                // hnsw_rs may return fewer than k_eff if the index is tiny;
                // leave remaining slots at the default padding.
                for (i, n) in nbrs.iter().enumerate().take(k_eff) {
                    let within_idx = n.d_id;
                    idx_row[start + i] = cells[within_idx];
                    dist_row[start + i] = n.distance;
                }
            });
    }

    BbknnResult {
        indices,
        distances,
        batches,
        k_per_batch,
        k_total,
    }
}

/// Compute neighbor-set recall of `got` (e.g. hnsw output) against an
/// exact ground truth `truth` (e.g. brute-force). Both arrays must share
/// the same `(n_cells, k_total)` layout. Returns mean per-cell recall
/// over `n_cells`, ignoring `u32::MAX` padding slots.
pub fn neighbor_recall(truth: &[u32], got: &[u32], k_total: usize) -> f64 {
    assert_eq!(truth.len(), got.len());
    let n_cells = truth.len() / k_total;
    let mut total = 0.0_f64;
    let mut counted = 0_usize;
    for c in 0..n_cells {
        let t_row = &truth[c * k_total..(c + 1) * k_total];
        let g_row = &got[c * k_total..(c + 1) * k_total];
        let t_set: std::collections::HashSet<u32> =
            t_row.iter().copied().filter(|&v| v != u32::MAX).collect();
        if t_set.is_empty() {
            continue;
        }
        let hits = g_row
            .iter()
            .copied()
            .filter(|v| *v != u32::MAX && t_set.contains(v))
            .count();
        total += hits as f64 / t_set.len() as f64;
        counted += 1;
    }
    if counted == 0 {
        0.0
    } else {
        total / counted as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn two_batches_small() {
        // 6 cells, 2D, two batches of 3 each.
        // Batch 0 cells: 0, 1, 2 at x=0, 0.1, 5 (y≈0)
        // Batch 1 cells: 3, 4, 5 at x=0, 0.2, 5.1 (y≈0.1)
        let pca: Array2<f32> = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 0.0],
            [0.0, 0.1],
            [0.2, 0.0],
            [5.1, 0.1],
        ];
        let batch: Array1<i32> = array![0, 0, 0, 1, 1, 1];
        let res = bbknn_brute(pca.view(), batch.view(), 2);
        assert_eq!(res.batches, vec![0, 1]);
        assert_eq!(res.k_per_batch, 2);
        assert_eq!(res.k_total, 4);

        // Cell 0 — batch-0 nearest 2: self (dist 0) and cell 1 (dist 0.1).
        // Batch-1 nearest: cell 3 (dist 0.1) and cell 4 (dist 0.2).
        let row0 = &res.indices[0..4];
        assert_eq!(row0, &[0u32, 1, 3, 4]);
        let d0 = &res.distances[0..4];
        assert!(d0[0] < 1e-6, "self-dist = {}", d0[0]);
        assert!((d0[1] - 0.1).abs() < 1e-5, "dist = {}", d0[1]);
    }

    #[test]
    fn batch_smaller_than_k_padded() {
        // Batch 1 has 1 cell; k=2 → second slot is u32::MAX.
        let pca: Array2<f32> = array![[0.0], [1.0], [10.0]];
        let batch: Array1<i32> = array![0, 0, 1];
        let res = bbknn_brute(pca.view(), batch.view(), 2);
        assert_eq!(res.batches, vec![0, 1]);
        assert_eq!(res.k_total, 4);

        // Cell 2 is in batch 1 alone → batch-1 slots [MAX, -]; slot 0 is 2
        // (itself at distance 0), slot 1 is padding.
        let row2_b1 = &res.indices[2 * 4 + 2..2 * 4 + 4];
        assert_eq!(row2_b1[0], 2);
        assert_eq!(row2_b1[1], u32::MAX);
        let d2_b1 = &res.distances[2 * 4 + 2..2 * 4 + 4];
        assert!(d2_b1[0] < 1e-6);
        assert!(d2_b1[1].is_infinite());
    }

    #[test]
    fn deterministic_across_runs() {
        // Same input should yield bit-identical output.
        let pca: Array2<f32> = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let batch: Array1<i32> = array![0, 0, 0, 0, 1, 1, 1, 1];
        let r1 = bbknn_brute(pca.view(), batch.view(), 3);
        let r2 = bbknn_brute(pca.view(), batch.view(), 3);
        assert_eq!(r1.indices, r2.indices);
        assert_eq!(r1.distances, r2.distances);
    }

    #[test]
    fn hnsw_recall_vs_brute_matches_well() {
        // 300 cells, 3 batches, 10 dims. HNSW recall of brute top-3 should
        // be >= 0.9 at ef_search = 32.
        use ndarray::{Array1, Array2};
        let n = 300;
        let d = 10;
        let mut data = Vec::with_capacity(n * d);
        for i in 0..n {
            for j in 0..d {
                // Deterministic pseudo-random — avoid rand dep in tests
                let x = ((i * 131 + j * 17) as f32).sin();
                data.push(x);
            }
        }
        let pca = Array2::from_shape_vec((n, d), data).unwrap();
        let batch = Array1::from_iter((0..n).map(|i| (i % 3) as i32));
        let k = 3;

        let brute = bbknn_brute(pca.view(), batch.view(), k);
        let hnsw = bbknn_hnsw(pca.view(), batch.view(), k, 32);

        let recall = neighbor_recall(&brute.indices, &hnsw.indices, brute.k_total);
        assert!(recall >= 0.9, "HNSW recall = {:.3} < 0.9", recall);
    }

    #[test]
    fn single_batch_equals_plain_knn() {
        // With one batch, BBKNN == regular k-NN.
        let pca: Array2<f32> = array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0],];
        let batch: Array1<i32> = array![0, 0, 0, 0, 0];
        let res = bbknn_brute(pca.view(), batch.view(), 3);
        assert_eq!(res.batches, vec![0]);
        // Cell 0's 3-NN: self (0), cell 1 (0.5), cell 2 (1.0).
        let row0 = &res.indices[0..3];
        assert_eq!(row0, &[0u32, 1, 2]);
    }
}
