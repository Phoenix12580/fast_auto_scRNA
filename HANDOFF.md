# 交接文档 — 新会话起点

**生成日期**：2026-04-26
**最新 commit**：`8331f8e` (origin/main)
**当前阶段**：上游 v2-P12 完结，下游分析待启 (v2-P13)

---

## 1. 当前状态（一句话）

上游单细胞流水线（QC → 集成 → CHAMP 聚类 → scIB）全部完成；222k 前列腺图谱
跑完封装在 `F:\down_ana\2026-04-26_atlas_222k_v2p12\`；下一步是**下游生物学
分析 + Rust 加速**，路线图见 [`DOWNSTREAM_ROADMAP.md`](DOWNSTREAM_ROADMAP.md)。

---

## 2. 项目结构

```
F:\fast_auto_scRNA_v2\                              代码仓库（git main）
├── README.md                                        项目主入口（中文，已同步到 v2-P12）
├── INSTALL.md                                       WSL/Windows 安装步骤
├── ROADMAP.md                                       完整里程碑历史 + 上游待办
├── DOWNSTREAM_ROADMAP.md                            下游分析 3 阶段计划（Stage 1-3）
├── HANDOFF.md                                       本文档
│
├── pyproject.toml                                   依赖 + scvi 可选 extra
├── fast_auto_scrna/                                 Python 包
│   ├── config.py                                    PipelineConfig dataclass（所有旋钮）
│   ├── runner.py                                    run_from_config 主入口 + Phase 1/2a/2b/2c 编排
│   ├── io/                                          h5ad / rda / Seurat 加载 + QC
│   ├── preprocess/                                  normalize / HVG / scale
│   ├── pca/                                         随机化 PCA + Gavish-Donoho（Rust 后端）
│   ├── integration/                                 默认 3 路（bbknn / harmony / fastmnn）+ scvi opt-in
│   ├── neighbors/                                   kNN + fuzzy_simplicial_set
│   ├── scib_metrics/                                iLISI/cLISI/GC/3×ASW/SCCAF
│   ├── umap/                                        UMAP layout
│   ├── cluster/                                     Leiden + CHAMP picker (Weir 2017)
│   │   ├── leiden.py
│   │   ├── champ.py                                 CHAMP 算法（凸包 + 模块度）
│   │   └── resolution.py                            auto_resolution 入口 + plot_champ_curve
│   ├── rogue/                                       Rust ROGUE
│   ├── plotting/                                    所有图（dual PDF+PNG，rasterized 散点）
│   │   ├── comparison.py                            cross-route 对比 + emit_route_plots
│   │   └── __init__.py
│   ├── downstream/                                  下游分析（v2-P13+，骨架已建）
│   │   ├── deg/                                     Stage 1 — Wilcoxon DEG
│   │   ├── annotate/                                Stage 2 — marker-panel scoring
│   │   │   └── panels/                              JSON/YAML marker panels
│   │   ├── gsea/                                    Stage 3 — pathway enrichment
│   │   └── plotting/
│   ├── _native/                                     Rust PyO3 bindings (maturin built .so/.pyd)
│   └── common/                                      共享工具
│
├── rust/                                            Rust workspace
│   ├── Cargo.toml
│   └── crates/
│       ├── kernels/                                 纯 Rust 算法（pca/bbknn/harmony/fuzzy/metrics/rogue/silhouette/umap）
│       └── py_bindings/                             PyO3 wrapper → fast_auto_scrna._native
│
├── tests/                                           pytest（13 passed + 4 scvi-skip）
│   └── test_smoke.py                                合成 500 cell 端到端 + scIB 并行等价性
│
├── benchmarks/                                      微基准 + 对比 + 重生成脚本
│   ├── smoke_222k.py                                222k 全管线 smoke
│   ├── smoke_pancreas.py                            1000 cell smoke
│   ├── bench_phase2b_champ_222k.py                  CHAMP vs knee 微基准
│   ├── regen_v2p12_deliverable_plots.py             重生成 dual-format 全套图
│   ├── plot_champ_vs_knee_222k.py                   CHAMP UMAP 对比（cache-aware）
│   ├── validate_scib_parallel_222k.py               scIB 并行验证（已弃，留作记录）
│   ├── inspect_atlas_h5ad.py                        h5ad metadata dump
│   └── out/                                         所有 bench 输出（.gitignore）
│       └── smoke_222k_v2p10.h5ad                   3.9 GB 缓存 atlas（含 4 路 CHAMP）
│
└── data/                                            原始数据软链（.gitignore）
    ├── pancreas_sub.rda → F:\NMF_rewrite\...
    └── StepF.All_Cells.h5ad → F:\NMF_rewrite\...

F:\down_ana\                                         交付包根目录
├── 2026-04-26_atlas_222k_v2p12\                    上游 v2-P12 交付（已封装）
│   ├── METADATA.md                                  完整说明 + picker per-route 表
│   ├── data\atlas_222k_v2p12.h5ad                  3.9 GB（4 路集成 + 4 路 CHAMP 全聚类）
│   ├── plots\                                       10 plots × 2 (PDF+PNG) = 20 files, 6.8 MB
│   │   ├── 01_integration_umap_comparison.{pdf,png}
│   │   ├── 02_scib_heatmap.{pdf,png}
│   │   ├── 03_rogue_per_cluster.{pdf,png}
│   │   ├── 04_champ_landscape_grid.{pdf,png}      4 路 CHAMP 凸包对比
│   │   ├── 05_picker_umap_per_route.{pdf,png}     knee vs CHAMP × 4 路
│   │   └── champ_landscape_<route>.{pdf,png} × 4
│   └── logs\                                        5 个 bench/run log
│
└── 2026-04-26_downstream\                          下游分析待填充
    ├── 01_deg\plots\                                Stage 1 输出位
    ├── 02_annotate\plots\                           Stage 2 输出位
    └── 03_gsea\plots\                               Stage 3 输出位
```

---

## 3. 上次会话发生了什么（最近 8 个 commit）

```
8331f8e feat(plotting): dual PDF+PNG output + cross-route comparison plots
5d53b52 docs+chore: downstream analysis roadmap + module skeleton
5114805 docs(README): sync to v2-P12 — CHAMP picker, 4-route integration, WSL workflow
74e8dfd feat(v2-P12): CHAMP — Convex Hull of Admissible Modularity Partitions (Weir 2017)
711223f perf(v2-P12): skip Phase 2b duplicate scIB recompute (~9 min saved on 222k)
c886482 docs(ROADMAP): v2-P11 result + ASW direction pivot to GPU silhouette
641708f feat(v2-P11): Phase 2a parallel scIB scaffolding — default off (BLAS-bound, no win on 16-core)
88a5213 chore: scvi-tools optional + WSL-friendly setup
```

**净收益**：222k all-mode wall **100 → 68 min（−32%）**，picker 从启发式
knee 改为 paper-cited CHAMP（Weir 2017 凸包模块度法）。

---

## 4. 下一阶段：下游分析（v2-P13 起）

详见 `DOWNSTREAM_ROADMAP.md`。3 阶段：

| Stage | Milestone | 内容 | 工作量 | Rust 加速 |
|---|---|---|---|---|
| **1** | v2-P13 | DEG / Marker (Wilcoxon U-test) | 2-3 天 | 复活 v1 wilcoxon.rs，预期 50-100× |
| 2 | v2-P14 | 细胞类型自动注释（marker panel scoring） | 1-2 天 | 无（逻辑层） |
| 3 | v2-P15 | GSEA / pathway 富集 | 3-5 天 | Rust permutation kernel，预期 30-50× |

**Stage 1 起步建议**（沿用「Python 先 reference 跑通 → Rust 加速 → 验证一致」
模式，沉淀的 3 条记忆 cover 这个）：
1. `fast_auto_scrna/downstream/deg/wilcoxon.py` —— 包装 `scanpy.tl.rank_genes_groups`
2. 在 `atlas_222k_v2p12.h5ad` 上跑一遍 8 cluster 的 marker
3. 输出到 `F:\down_ana\<date>_downstream\01_deg\` (CSV + dual-format heatmap)
4. **stop and review** 看输出合理再 Rust 化
5. Rust 加速 + 数值对照 + commit

---

## 5. 新会话 quick start

```bash
# WSL（默认）
cd /mnt/f/fast_auto_scRNA_v2
git pull --ff-only
source .venv-wsl/bin/activate

# 用缓存的 222k atlas 起步（已含 4 路 CHAMP labels + curves）
python -c "import anndata as ad; a = ad.read_h5ad('benchmarks/out/smoke_222k_v2p10.h5ad', backed='r'); print(a)"

# 或直接读交付版（同内容）
python -c "import anndata as ad; a = ad.read_h5ad('/mnt/f/down_ana/2026-04-26_atlas_222k_v2p12/data/atlas_222k_v2p12.h5ad', backed='r'); print(a)"

# 跑测试
pytest tests/test_smoke.py -v
```

如果环境从零开始（新机器），见 `INSTALL.md` 完整步骤。

---

## 6. 关键决策 / 全局原则（沉淀到记忆）

复用上游教训，下游也按这套来：

1. **Python 主体 + Rust 内核**：算法 Python，重计算 Rust（pyo3）
2. **Cache-first**：所有产物写回 h5ad，下次免重算（已有 `champ_curve_<route>` /
   `leiden_<route>_champ` 等）
3. **微基准前置**（沉淀的 3 条记忆 cover）：
   - `feedback_rust_speedup_assumption.md`
   - `feedback_blas_bound_multiprocess.md`
   - `feedback_two_stage_knee_picker.md`
4. **Validation-driven**：先 reference impl 拿 ground truth，再 Rust 化对照
5. **Dual-format plots**：所有图 PDF+PNG，散点 `rasterized=True`（AI 不卡）
6. **每阶段独立可发**：dict 接口，互不耦合

**结果一致性是硬约束** —— 任何 picker / metric 替换都必须在真数据 atlas 上
验证 |Δ| 在可接受范围（典型阈值：picked_r ±0.05，Δk ≤ 2，metric ≤ 1e-4）。

---

## 7. 已知技术债 / 未触碰的项

- WSL 的 `/mnt/f/` 缓存有时和 Windows 写入不同步（Bash tool 写 vs PowerShell
  写互不可见）—— delivery 时用 PowerShell-side 的 `Copy-Item` 最稳
- scVI 训练 21.5 min 是 PyTorch GPU bound，能动空间小（已经压到 36 epochs）
- ASW 的 GPU silhouette pivot（torch.cuda）在 ROADMAP 里待启 —— 可拉 22 min
  → ~1 min，但需 WSL 装 CUDA torch（cudnn 600 MB 下载之前失败过）
- 旧版 bench scripts（pancreas / stability_picker / first_plateau_designer
  等）部分引用已删的 optimizer 函数，未清理（不影响主管线）

---

## 8. 联系点 / 如何索引知识

| 想找什么 | 看哪里 |
|---|---|
| 上游全管线参数 | `fast_auto_scrna/config.py` |
| 上游性能 baseline | `ROADMAP.md` 的 "v2-P12 实测" 段落 + `benchmarks/out/smoke_222k_v2p10.log` |
| CHAMP 数学/算法 | `fast_auto_scrna/cluster/champ.py` 模块 docstring |
| 下游计划 | `DOWNSTREAM_ROADMAP.md` |
| 历史教训 | `C:\Users\Administrator\.claude\projects\F--NMF-rewrite-fast-auto-scRNA\memory\` |
| 上游交付物 | `F:\down_ana\2026-04-26_atlas_222k_v2p12\` |

新窗口启动时：先读 README → ROADMAP "当前里程碑" 段 → DOWNSTREAM_ROADMAP，
然后就可以接手干活了。
