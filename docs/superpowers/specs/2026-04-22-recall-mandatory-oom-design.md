# recall 作为对外 pipeline 必备步骤 + OOM 通过 anndata-oom 解决

**日期**: 2026-04-22
**作用域**: `fast_auto_scRNA_v1`(对外 pipeline v1,后续推 github.com/Phoenix12580/fast_auto_scRNA)+ `scvalidate_rewrite/scvalidate/recall_py/`(底层库)
**状态**: 设计已定,M0 起手姿势待用户确认

**v1 目的**:在 `fast_auto_scRNA_v1/` 目录完成"recall 必备 + anndata-oom OOM 方案 + 对比报告",通过端到端 benchmark 后打 v1 tag 推 GitHub。旧 `fast_auto_scRNA/` 保留作 baseline 参照。

---

## 1. 背景

### 1.1 现状

`fast_auto_scRNA` 的对外 pipeline 目前对"cluster 数该定多少"有两条路径,并存但语义不一致:

1. **现用的"自动选 k"**(在 `PipelineConfig` 里):扫 `leiden_resolutions=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0]`,取 k 落在 `leiden_target_n=(8, 30)` 范围内的**最小 resolution**。这是启发式,**不做任何统计校准**。
2. **recall**(`scvalidate/recall_py/find_clusters_recall`):knockoff-filter 驱动的迭代降 resolution 法,每轮检查是否存在"两簇在差异表达上不可区分";有则 `res *= 0.8` 继续,无则收敛。统计上严谨。

**`run_recall` 默认 False**(`README.md:243`),注释写 "≤ 10k 建议开";场景 C(atlas 100k+)注释 "`recall O(K²·N) 爆`" 强制关 — v1 要把这个参数**直接删除**,所有数据全自动跑 recall。

### 1.2 要解决的三件事

1. **recall 从"可选"升级为"必备"** — 所有数据都过 recall,不给 opt-out。
2. **输出 recall vs baseline 自动选 k 的对比** — pipeline 结果里强制包含"target_n 规则给的 k"和"recall 校准给的 k",让用户看到校准前后的差异。
3. **破解 157k 上的 OOM** — 当前 v0.4 Rust recall 在 157k 外推 ~90 GB 峰值,超出 WSL 64 GB 上限(`memory: project_scvalidate_v04_progress.md`)。

---

## 2. 选定方案:anndata-oom 全路径

### 2.1 为什么是 anndata-oom

- recall 90 GB 的来源三块都能被 anndata-oom 的 chunked/lazy API 覆盖:augmented int32(20.5 GB)、log_counts f32(20.5 GB)、scanpy preprocess 瞬时 copies(25-40 GB)。
- 用户 memory(`reference_anndata_oom.md`)已明确定为 OOM 首选方案。
- WSL 是主力开发环境,Linux wheel 开箱即用;Windows wheel 的问题不影响。
- 157k 预估峰值 **2.5 GB**(按 README PBMC 8k 27.8× 收益外推),同一机器还能往上扛到 1M 级。

### 2.2 拒掉的替代方案

| 方案 | 为什么不选 |
|---|---|
| Subsample(random stratified floor=10k) | OOM 能解,但改"全量 recall"的统计语义,需要后面再用 metacell 升级。用户明确拒绝 |
| HVG-first + int16 + preprocess-out-of-loop 组合 | 157k peak 压到 20-25 GB 塞进 64 GB,但要 1 周,上限 ~300k;anndata-oom 更快更干净 |
| 128 GB 内存升级 | 生产服务器约束不确定;代码方案优先 |
| GPU offload | 生产无 GPU,排除 |

---

## 3. 架构与组件

### 3.1 数据流

```
┌────────────────────────────────────────────────────────────────────┐
│ fast_auto_scRNA outer pipeline                                      │
│                                                                     │
│   load → QC → lognorm → HVG → scale → PCA                           │
│        ↓                                                            │
│   [integration route(s)]                                            │
│        ↓                                                            │
│   neighbors → UMAP                                                  │
│        ↓                                                            │
│   ── Leiden 扫 resolutions ────────────────┐                        │
│        ↓                                    │                        │
│   baseline k = target_n 规则选出的          │  ← 原 baseline         │
│   {resolution*, k_baseline, labels_baseline}│                        │
│                                             │                        │
│   ── recall(必备, anndata-oom 路径)───────┤                        │
│        ↓                                    │                        │
│   knockoff → oom-backed augmented h5ad →    │                        │
│   lazy preprocess → PCA → neighbors(一次)  │                        │
│        ↓                                    │                        │
│   while iter:                               │                        │
│     leiden(res) → per-pair wilcoxon         │                        │
│     (chunked Rust) → knockoff filter        │                        │
│     merged? res *= 0.8 : break              │                        │
│        ↓                                    │                        │
│   {resolution_recall, k_recall, labels_recall, k_trajectory}        │
│                                                                     │
│        ↓         ↓                                                  │
│   ── comparison report(必备输出)───────────────────────────────── │
│   RecallComparisonReport:                                           │
│     k_baseline, k_recall, Δk,                                       │
│     ARI(labels_baseline, labels_recall),                            │
│     n_clusters_merged_by_recall, per_baseline_cluster_fate          │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 改动清单(按模块)

#### 模块 A — `scvalidate_rewrite/scvalidate/recall_py/core.py`

**接口变更**:`find_clusters_recall` 第一参数从 `counts_gxc: np.ndarray | sparse` 改成 `adata_backed: oom.AnnDataOOM | AnnData`。保留对 dense ndarray 的接受,内部检测后决定是否要先 `write → oom.read` 走 backed 路径。

**内部重构**:
1. knockoff 生成 → `_write_augmented_h5ad(adata, knock, path, chunk_cells=5000)`,通过 `adata.write(path)` 的 chunked write 落盘,peak RAM ≈ 1 chunk。
2. `aug_log = oom.log1p(oom.normalize_total(aug))` — 纯 lazy,不入内存。
3. HVG、scale、chunked PCA、neighbors **全部移到 while 循环外**(原实现每轮都 re-preprocess,内存翻倍、浪费 CPU)。
4. while 循环里只调用 `sc.tl.leiden(adata, resolution=r)` 改 labels。
5. 新增 `_wilcoxon_pair_chunked(aug_log, n_real, mask1, mask2, chunk=2000)`:按 gene 轴分块从 `aug_log.X.chunked(2000)` 拉 chunk,每个 chunk 调既有的 `scvalidate_rust.wilcoxon_ranksum_matrix`,concat p 值。Rust kernel 不动。

**`RecallResult` 扩展**:增字段
```python
@dataclass
class RecallResult:
    labels: np.ndarray
    resolution: float
    n_iterations: int
    per_cluster_pass: dict[int, bool]
    # 新增
    resolution_trajectory: list[float]     # 每轮的 resolution
    k_trajectory: list[int]                # 每轮的 k
    converged: bool                        # False 表示被 max_iterations 截停
```

**Scratch 清理**:`/tmp/aug_<uuid>.h5ad` 用 `tempfile.TemporaryDirectory()` 包住,退出或异常都删。

#### 模块 B — `fast_auto_scRNA/scatlas_pipeline` 接入

**新增** `RecallComparisonReport` dataclass:
```python
@dataclass
class RecallComparisonReport:
    k_baseline: int                        # target_n 规则选的 k
    resolution_baseline: float             # 对应 resolution
    k_recall: int                          # recall 收敛的 k
    resolution_recall: float               # recall 最终 resolution
    delta_k: int                           # k_baseline - k_recall (正值=校准后更少)
    ari_baseline_vs_recall: float          # 分区一致性
    recall_converged: bool                 # 是否真收敛(vs max_iterations 截停)
    per_baseline_cluster_fate: dict[int, str]  # "kept" / "merged_with_X" / "split_into_[X,Y]"
    k_trajectory: list[int]                # recall 每轮 k,画轨迹图
    recall_wall_time_s: float
```

**API 变更**(`PipelineConfig`):
- `run_recall` 参数**直接删除**。recall 是 v1 对外 pipeline 的自动化卖点,不允许 opt-out。用户传 `run_recall=...` 时触发 `TypeError: unexpected keyword argument 'run_recall' (removed in v1: recall is mandatory)`,附迁移提示。
- 新增 `recall_max_iterations=20`(原 scvalidate 默认)和 `recall_fdr=0.05`,从 scvalidate 透传。
- 新增 `recall_scratch_dir: Path | None = None`(None → 用 `tempfile.TemporaryDirectory()` 默认路径)。

**pipeline 输出新加**:
- `adata.uns["recall_comparison"] = asdict(report)`
- `out/<run_id>_recall_comparison.json`
- `out/<run_id>_recall_k_trajectory.png`(resolution vs k 折线,标注收敛点)

#### 模块 D — `fast_auto_scRNA_v1/` 目录结构

v1 作为新工作区与 github 推送目标,起手姿势**待 M0 决策**(见 §8):

```
F:/NMF_rewrite/fast_auto_scRNA_v1/
├── docs/
│   └── superpowers/specs/  ← 本 spec 所在
├── scatlas/                ← 来源于 fast_auto_scRNA/scatlas
├── scatlas_pipeline/       ← 集成入口,v1 改动主场
├── scvalidate_rewrite/     ← recall_py 必改;rust kernel 保持
├── README.md               ← 改:recall 从"可选"升格为"必备自动化";加 comparison 段
├── ROADMAP.md              ← 更新 recall 状态
├── INSTALL.md              ← 加 anndataoom 依赖说明
└── tests/                  ← 加 3 套新测试
```

最终通过 `git remote add origin https://github.com/Phoenix12580/fast_auto_scRNA.git` 并推 `v1` 分支(或 tag)。

#### 模块 C — 兜底:当 anndata-oom 没装

安装 gate:`scvalidate.recall_py` 加载时 `try: import anndataoom`,失败时 `find_clusters_recall` 回退到原 dense 路径**并打印明确 warning**:"anndataoom 未安装,recall 走 dense 路径,>30k 可能 OOM"。不 hard-require,避免影响小数据用户。

---

## 4. 测试方案

### 4.1 已有测试复用

- `tests/test_recall_core.py` — dense 路径必须保持通过(兜底不坏)
- `tests/test_rust_wilcoxon_parity.py` — Rust kernel 不动,parity 不受影响

### 4.2 新增

1. **`test_recall_oom_parity.py`**:同一 10k epithelia 数据,dense 路径 vs oom 路径的 `labels` ARI ≥ 0.95、`k_recall` 差 ≤ 1、`resolution` 相对差 < 5%。(允许 chunked PCA 的随机 SVD 微小扰动,不做 bit-identical)
2. **`test_recall_oom_memory.py`**:peak RSS < 8 GB 在 50k 子样本上(dense 路径参考 34 GB),断言至少 4× 节省。
3. **`test_recall_comparison_report.py`**:pipeline 端到端 pancreas_sub 1k 跑完,`adata.uns["recall_comparison"]` 存在且字段齐全。
4. **Benchmark 新增** `benchmark/epithelia/recall_oom_157k.py`:WSL 全量 157k 跑通,目标 peak RAM < 5 GB、wall 含首次 I/O < 50 min(subsample 那条路径作为对比点在 README 里注明)。

### 4.3 验收标准

- 10k epithelia dense-vs-oom ARI ≥ 0.95 ✓
- 157k 全量能跑完(当前 OOM)✓
- `RecallComparisonReport` 在所有规模(1k/10k/157k)都产出 ✓
- `tests/` 36/36 通过 + 新增 3 套全绿 ✓

---

## 5. 非目标(明确不做)

1. **Subsample / metacell 路径**:短期不做;用户原话"不了"拒绝。Metacell 留给未来 `fast_auto_scRNA` 的 metacell+Harmony2 pipeline 落地后再议。
2. **HVG-first knockoff**:anndata-oom 让内存问题本身不存在,这条优化没必要现在做。
3. **GPU recall**:生产环境约束排除。
4. **metacell 级 recall**(即 recall 跑在 metacell 聚合后的 super-cells 上):等 metacell pipeline 先落地,属于下一个 spec。
5. **scSHC 的 OOM 路径**:超范围,scSHC 有独立进度(`feedback_drop_scshc.md` 已决策不 Rust 化)。

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| scratch h5ad 20 GB 占 SSD;崩溃没清 | `TemporaryDirectory` + `atexit` 双保险;`/tmp` 不够时允许配 `recall_scratch_dir` |
| chunked PCA 随机 SVD 非确定性扰动 leiden 分区 | parity 验收用 ARI ≥ 0.95 而非 bit-identical;写 test 固定 `random_state` |
| anndata-oom 版本升级破坏 API | `pyproject.toml` pin 到 tested 版本,CI 跑 oom 路径 |
| 首次 I/O 40s 开销让 10k 小数据变慢 | 数据 < 30k 时走 dense 路径(`if n_cells < 30_000: dense else: oom`);阈值 30k 是内存/I/O 交叉点,实测可调 |

---

## 7. 里程碑与工作量

| 阶段 | 工作 | 估时 |
|---|---|---|
| M0 | 初始化 `fast_auto_scRNA_v1/` 工作区(A 或 B,见 §8)+ git 初始化 + anndataoom 依赖装好 | 0.5 天 |
| M1 | `recall_py/core.py` 接 backed AnnData,dense 路径兼容保留 | 1 天 |
| M2 | `_wilcoxon_pair_chunked` + `RecallResult` 扩展 + scratch 清理 | 1 天 |
| M3 | `RecallComparisonReport` + fast_auto_scRNA_v1 pipeline 集成 + 删 `run_recall` 参数 | 1 天 |
| M4 | 新增 3 套测试 + 10k parity 验收 | 0.5 天 |
| M5 | 157k WSL 全量 benchmark + README 更新 | 0.5 天 |
| M6 | `git push origin v1` 到 github.com/Phoenix12580/fast_auto_scRNA | 0.5 天 |

**总计 ≈ 5 天**。

---

## 8. 已确认决策 + 待定 M0 起手姿势

### 已确认
1. **`run_recall` 直接删除**(2026-04-22 用户决策):v1 核心是自动化,recall 不给 opt-out。传参触发 TypeError。
2. **30k 阈值**:<30k 走 dense,≥30k 走 oom。接受。
3. **整合目标**:v1 与 github.com/Phoenix12580/fast_auto_scRNA 整合,推送为 v1 分支/tag。
4. **scratch/report 命名**:默认采用 `recall_scratch_dir` + `RecallComparisonReport`,后续如有更好命名可改,不阻塞落地。

### M0 待定:v1 工作区的起手姿势

已查 `fast_auto_scRNA/`:on `main` 与 origin 同步,工作树干净。已有 commit `d584627` 装了 `use_anndataoom=False` 钩子(仅 import,未进 recall 路径)。

三个选项:

| 选项 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| A. 全量复制 | `cp -r fast_auto_scRNA/ fast_auto_scRNA_v1/` | baseline 完整可跑 | 磁盘翻倍;脱离 git,push 麻烦 |
| B. 重新 clone | `git clone ... fast_auto_scRNA_v1` + `git checkout -b v1` | 干净血统 | 多一份 .git 对象 ~100 MB |
| **C. git worktree(推荐)** | `cd fast_auto_scRNA && git worktree add ../fast_auto_scRNA_v1 -b v1` | 共享 `.git`,零存储开销;v1 分支原生存在;两个工作目录同时可用,baseline 永远在 main | 小众概念,`git worktree list` 要记 |

**推荐 C**:一条命令建立平行工作区 + v1 分支,既满足"新建文件夹"又满足"推 GitHub",后续 `git push origin v1` 直发。

命令(等用户确认后执行):
```bash
cd F:/NMF_rewrite/fast_auto_scRNA
git worktree add ../fast_auto_scRNA_v1 -b v1
# 之后所有 v1 改动在 fast_auto_scRNA_v1/ 里做,git commit 记到 v1 分支
# ship 前: git push origin v1
```

**注意**:本 spec 当前位于 `fast_auto_scRNA_v1/docs/superpowers/specs/`,但这个目录现在还**没被 git 追踪**(v1 worktree 未建)。执行 worktree 命令前要先把 spec 移出或 stash,命令完成后再移回,不然 `git worktree add` 会报 "already exists"。处理步骤:

```bash
mv F:/NMF_rewrite/fast_auto_scRNA_v1 F:/NMF_rewrite/.tmp_v1_spec
cd F:/NMF_rewrite/fast_auto_scRNA
git worktree add ../fast_auto_scRNA_v1 -b v1
mkdir -p ../fast_auto_scRNA_v1/docs/superpowers/specs/
mv ../.tmp_v1_spec/docs/superpowers/specs/*.md ../fast_auto_scRNA_v1/docs/superpowers/specs/
rm -rf ../.tmp_v1_spec
cd ../fast_auto_scRNA_v1 && git add docs/ && git commit -m "docs: recall mandatory + OOM design spec"
```
