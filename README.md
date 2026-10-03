# Foundation Models Interpretability

Gene-level interpretability of two single-cell foundation models, **scGPT** and **Geneformer (V2, 104M)**, for Parkinson's disease (PD) vs control, on two datasets and four cell types (astrocytes, DA neurons, microglia, oligodendrocytes).

For every (model, dataset, cell type) the pipeline ranks genes with three methods, cuts each ranking into a gene set with three selection rules, and evaluates every gene set with GO enrichment and GO semantic similarity to PD reference panels:

| Dimension | Values |
|---|---|
| Ranking | pooled attention difference (PD - control), CSSI-max attention, OLS-residualized attention |
| Selection (top-N) | top-30, z-score, elbow |
| Model | scGPT, Geneformer (zero-shot; the LoRA fine-tuned models are also available, top-30 only) |
| Dataset | A, B |
| Cell type | astrocytes, DA neurons, microglia, oligodendrocytes |

## Environment

```
python -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

Everything runs in this one environment: Geneformer is a plain BERT loaded with `transformers`/`peft` and its cells come already tokenized, so the `geneformer` package is not needed. `go3` (GO semantic similarity, Rust) installs from PyPI as a prebuilt wheel. scGPT's flash-attention backend needs a CUDA GPU for the model tasks; the OLS and post-hoc tasks run on CPU.

## Data

`data/` is not in git. Expected layout:

```
data/
├── scGPT_brain/                 scGPT zero-shot pretrained model (args.json, vocab.json, best_model.pt)
├── Geneformer_V2_104M/          Geneformer zero-shot pretrained model + token_dictionary_gc104M.pkl
├── go3_refs/                    GO ontology (go-basic.obo) and human GO annotations (goa_human.gaf)
├── pd_panels/                   PanelApp "Parkinson Disease and Complex Parkinsonism" panel and
│                                ParkinsonsUK-UCL GO annotations (QuickGO)
├── dataset_a/
│   ├── mixed_split_scgpt/       scGPT cells: normalised + log1p expression of 1000 HVG (*_train/val/test.h5ad)
│   ├── mixed_split_geneformer/  Geneformer cells, already tokenized (*_train/val/test.dataset),
│   │                            and gene_names.csv (token -> gene symbol)
│   ├── step5_1_lora_scgpt/      LoRA fine-tuned scGPT classifier (adapters merged, best_full_model.pt)
│   └── step5_1_lora_geneformer/ LoRA fine-tuned Geneformer classifier (best_lora/ + best_head.pt)
├── dataset_b/                   same layout as dataset_a
└── legacy_v0/                   inputs of the old Dataset A runs (not used by the pipeline)
```

In each dataset both models see the same cells and the same train / validation / test split (cell level, 80/10/10). Every cell has a `cell_type` and a binary `diagnosis`; the scGPT h5ad files also have a `donor` column (used only by `--permutation`). The pipeline always uses train + val + test together.

## Running

```
python run_tasks.py --task <task> --dataset <A|B> [options]
```

Each dataset writes to its own folder, `results/dataset_a/` or `results/dataset_b/`, so the post-hoc analyses never mix them.

### Model tasks (GPU)

| Task | What it does | Output folder |
|---|---|---|
| `attention_scgpt` | zero-shot scGPT: pooled attention, CSSI and their selections | `attention_zero_shot_scgpt/` |
| `attention_geneformer` | the same with zero-shot Geneformer | `attention_zero_shot_geneformer/` |
| `ig_lora_scgpt` | Integrated Gradients on the LoRA scGPT classifier (correctly classified cells), top-30 | `ig_lora_scgpt/` |
| `attention_lora_geneformer` | attention of the LoRA Geneformer classifier (correctly classified cells), top-30 | `attention_lora_geneformer/` |

### OLS tasks (CPU, after the attention tasks)

| Task | What it does | Output folder |
|---|---|---|
| `attention_residualized_ols_scgpt` | OLS residual of the scGPT pooled attention and its selections | `attention_residualized_ols_scgpt/` |
| `attention_residualized_ols_geneformer` | the same for Geneformer | `attention_residualized_ols_geneformer/` |

### Post-hoc tasks (CPU, after the tasks above)

Run in this order, or all at once with `--task postprocess`. `enrichment` (g:Profiler) and `gwas_similarity` (GWAS Catalog, cached after the first run) need internet access.

| Task | What it does | Output folder |
|---|---|---|
| `enrichment` | g:Profiler GO:BP/MF/CC, KEGG and Reactome enrichment of every gene set | `enrichment/` |
| `enrichment_summary` | term count, information content, intersection size and functional abundance per gene set, plus pairwise Mann-Whitney tests between the methods of each model (BH-adjusted) | `enrichment_summary/` |
| `gwas_similarity` | GO3 set-level similarity (lin/SimRel/wang, BMA) of every gene set to the PD/AD/CAD GWAS Catalog gene sets and the two PD panels | `gwas_similarity/` |
| `gene_matrix` | gene x gene GO3 similarity matrices vs the three PD references (GWAS, PanelApp, ParkinsonsUK-UCL) | `gene_matrix/` |
| `plots` | every comparison figure, one per model (also drawn by the tasks above) | `plots/` |

### Options

| Option | Meaning |
|---|---|
| `--task all` (default) | the four model tasks, then the two OLS tasks |
| `--dataset A\|B` | dataset (default A) |
| `--gpus 0,1` | model tasks only: one worker per GPU, whole cell types balanced by cell count |
| `--device cuda:0` | GPU for single-GPU runs |
| `--selection top30\|z-score\|elbow\|all` | gene-set selections used by the post-hoc tasks (default all) |
| `--permutation` | also run the donor-level permutation selector (scGPT only, slow) |
| `--donor-col` | donor column for `--permutation` (auto-detected, `donor` in both datasets) |
| `--max-cells-per-type N` | seeded random subset of at most N cells per cell type, for quick test runs; the OLS tasks must use the same N as the attention tasks |
| `--batch-scale auto\|N` | multiplier of the model batch sizes, which are tuned for an 8 GB GPU; `auto` (default) adds x1 per 10 GiB of GPU memory, up to x8 (RTX 2070 x1, L40 x4, H100 x7, H200 x8) |
| `--distances lin SimRel wang` | GO term similarities computed by `gwas_similarity` / `gene_matrix` (default all three; the plots use SimRel) |
| `--results-dir` | results root (default `results/`) |

Set `GO3_THREADS` to limit the CPU threads of the GO3 tasks (default: every core). Run with `python -u` (or `PYTHONUNBUFFERED=1`) when the output goes to a log file, so progress shows up as it happens.

The long loops (model forward passes, IG, Leiden resolutions, GO3 gene pairs, g:Profiler queries, gene matrices) show tqdm progress bars. When the output goes to a file, or with `--gpus` (several workers share one output), each bar is printed as one line every 30 s (`PROGRESS_INTERVAL` changes it) instead of being redrawn, so the logs stay readable; follow one task with e.g. `tail -f logs/B_attention_geneformer.log`.

### Run time

The GO3 similarity of `gwas_similarity` and `gene_matrix` dominates the post-hoc time: it grows with (genes in the gene sets) x (genes in the references, 1646 in total, 799 of them in the ParkinsonsUK-UCL panel), and the Geneformer z-score/elbow sets hold hundreds of genes. Every gene-to-gene similarity is therefore computed once per (ontology, distance) and cached in `results/go3_pair_cache/` (shared by both datasets, every selection and both tasks; it only depends on the GO files, delete it if `data/go3_refs/` changes). For the gene sets of a 200-cells-per-type run, the first post-hoc run is estimated at ~2.3 h on 24 cores with three distances and ~0.8 h with `--distances SimRel` (~9-11 h before the cache); later runs only compute pairs of genes not seen before.

### Full run

```
for D in A B; do
  python -u run_tasks.py --task attention_scgpt           --dataset $D --gpus 0,1
  python -u run_tasks.py --task attention_geneformer      --dataset $D --gpus 0,1
  python -u run_tasks.py --task ig_lora_scgpt             --dataset $D --gpus 0,1
  python -u run_tasks.py --task attention_lora_geneformer --dataset $D --gpus 0,1
  python -u run_tasks.py --task attention_residualized_ols_scgpt      --dataset $D
  python -u run_tasks.py --task attention_residualized_ols_geneformer --dataset $D
  python -u run_tasks.py --task postprocess --dataset $D
done
```

`scripts/run_full.sh` runs all of this unattended (the post-hoc tasks of A overlap with the model tasks of B), using every visible GPU (at most 4) and writing one log per task to `logs/`:

```
nohup bash scripts/run_full.sh > /dev/null 2>&1 &
tail -f logs/run_full.log
```

`PYTHON`, `GPUS`, `DISTANCES`, `GO3_THREADS`, `LOG_DIR` and `PROGRESS_INTERVAL` override its defaults (see the script header). `watch -n 30 bash scripts/progress.sh` shows the latest progress line of every running bar.

A quick end-to-end check before a full run: `--task all --max-cells-per-type 200`, then `--task postprocess --selection top30 --distances SimRel` (~20 min on 24 cores).

Measured on 2x RTX 2070 (8 GB) and 24 cores, a full run takes ~11-14 h: model tasks ~3.5 h for A and ~5.5-6 h for B (Geneformer and the LoRA tasks dominate; B's oligodendrocytes, 172k cells, keep one GPU busy the longest), post-hoc ~2.5 h for A and ~1.5 h for B with three distances. Larger GPUs shorten the model tasks; the post-hoc time depends on the CPU cores.

### Moving to another server

1. Clone the repository and copy `data/` (~2.5 GB, not in git), e.g. `rsync -a old-server:path/to/repo/data/ data/`. `results/go3_pair_cache/` can be copied too, to reuse GO3 similarities already computed.
2. Python 3.10 environment. Install torch from the PyTorch wheel index first, then the rest:
   ```
   pip install torch==2.3.0 torchtext==0.18.0 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
3. flash-attention 1.x (scGPT's attention backend) compiles from source against the local CUDA toolkit: `pip install flash-attn==1.0.4 --no-build-isolation`, with `nvcc` from a CUDA 12.x toolkit on the PATH (torch is built for CUDA 12.1). Ada (L40) runs Ampere kernels. On Hopper (H100, H200) the build must include sm_90 kernels; if scGPT fails with a "no kernel image" CUDA error, rebuild it with sm_90 added to the architectures in flash-attn's `setup.py`. Geneformer uses PyTorch SDPA and does not need it.
4. Check: `python -c "import flash_attn, go3, scgpt, transformers, peft"`, then the quick end-to-end check above on dataset A, then `scripts/run_full.sh`.

`visualize_cssi_embeddings.py --dataset <A|B>` draws the cell embeddings and Leiden strata CSSI used, per cell type, from the attention output (no inference).

## Methods

### Gene rankings

All attention scores use the CLS-token attention of one encoder layer (scGPT layer 11, Geneformer layer 10), computed from Q·Kᵀ of the CLS row, rank-normalised over each cell's genes and averaged over heads. One forward pass per cell type feeds every ranking.

- **Pooled attention** (`attention_zero_shot_<model>`): `mean_attention_pd - mean_attention_control` per gene. Each mean is over every cell of the condition, a cell without the gene counting as 0, so a gene seen in only a few cells cannot rank above one seen in all of them. The sign shows which condition attends to the gene more.
- **CSSI-max** (`attention_cssi_<model>`): the cells are split into ~6 strata by Leiden clustering (cosine k-NN graph, k = 15) of the residual-stream CLS embedding entering the attention layer; the same PD - control score is computed inside every stratum with at least 20 cells per condition, and each gene keeps its maximum. Without any valid stratum it falls back to the pooled score.
- **OLS-residualized** (`attention_residualized_ols_<model>`): across genes, the pooled score is regressed on three gene-level covariates of the same cells (mean expression, expression variance, dropout rate); the residual is the part of the attention not explained by how much / how often a gene is expressed.
  - scGPT: the covariates come from the h5ad expression matrix.
  - Geneformer: cells are rank-value encoded (no expression values), so the covariates use a rank pseudo-expression, `v = 1 - (p - 1) / (n - 1)` for the gene at position `p` of a cell's `n` genes (1 = the cell's most expressed gene) and `v = 0` when the gene is absent.

Fine-tuned (LoRA) variants, top-30 only:

- **Integrated Gradients** (`ig_lora_scgpt`): mean absolute attribution to the PD logit of the LoRA scGPT classifier over every correctly classified cell (0 where the gene is absent).
- **LoRA attention** (`attention_lora_geneformer`): the pooled attention score of the LoRA Geneformer classifier, on its correctly classified cells.

### Gene-set selections

Each zero-shot ranking is cut three ways (files `top30.csv`, `z-score.csv`, `elbow.csv`, columns `rank, gene, score`):

- **top30**: the 30 highest scores.
- **z-score**: genes with score > mean + 2.5 sd of all the gene scores. With ~17k genes, Geneformer z-score sets are much larger than scGPT's (1000 HVG).
- **elbow**: Kneedle elbow of the descending score curve (falls back to top-30 if there is no elbow).

`selection_summary.json` records the size and parameters of each set. With `--permutation`, a donor-level permutation selector (label permutations of donors, Cohen's d, BH q < 0.05 and |d| ≥ effect size) is also written (`permutation.csv`, `permutation_audit.csv`); it needs donor labels, so it is scGPT only.

### Gene-set metrics

- **Enrichment** (g:Profiler, significant terms only): number of terms, information content `-log2(term_size / domain_size)`, intersection size, functional abundance `intersection_size / term_size`.
- **GO semantic similarity** (GO3, BP/CC/MF, lin/SimRel/wang): set-level BMA similarity to the PD GWAS gene set (GWAS Catalog), the PanelApp PD panel (green genes) and the ParkinsonsUK-UCL GO-annotated genes, with Alzheimer and coronary artery disease GWAS sets as controls; gene x gene matrices vs the three PD references.

### Reproducibility

scGPT's binning breaks ties randomly, so tokenisation is seeded (`BINNING_SEED = 42` in `pipeline/config.py`); Leiden, the subsets of `--max-cells-per-type` and the permutations use seed 42. Results are identical across runs and across single- and multi-GPU runs on the same GPU model and `--batch-scale`. A different GPU or batch size changes the attention scores only by floating-point rounding (relative differences ~1e-4, from fp16 and padding), which can swap genes with nearly tied scores.

## Results layout

Inside `results/dataset_<a|b>/` (`<ct>` = cell type, `<model>` = `scgpt` / `geneformer`):

```
attention_zero_shot_<model>/<ct>/
├── top30_genes.csv              pooled ranking, top-30
├── csv/all_genes.csv            pooled ranking, every gene
├── csv/{top30,z-score,elbow}.csv + selection_summary.json
├── cssi_genes.csv               CSSI ranking, every gene
├── cssi/csv/{top30,z-score,elbow}.csv + selection_summary.json
├── cell_embeddings.npy, leiden_clusters.csv, cssi_metadata.json
attention_residualized_ols_<model>/<ct>/
├── residualized_genes.csv       score, OLS fit, residual and covariates of every gene
├── csv/ols.csv                  residual ranking, every gene
├── csv/{top30,z-score,elbow}.csv + selection_summary.json
└── ols_summary.json             R², coefficients, F-test
ig_lora_scgpt/<ct>/top30_genes.csv
attention_lora_geneformer/<ct>/top30_genes.csv
enrichment/[<selection>/]<method>/<ct>.csv, manifest.csv
enrichment_summary/[<selection>/]summary_table.csv, pairwise_tests.csv
gwas_similarity/refs/            reference gene sets (GWAS Catalog cached on first use)
gwas_similarity/[<selection>/]gene_level_similarity_all.csv
gene_matrix/[<selection>/]matrices/<ct>/<reference>__<distance>__<ontology>.csv, summary_statistics.csv
plots/<figure>_<model>[_<selection>].png
```

`results/go3_pair_cache/<ontology>__<distance>.npz` holds the cached gene-to-gene GO3 similarities shared by both datasets.

The top-30 outputs of the post-hoc tasks have no `<selection>/` subfolder. A method name in those outputs is `<ranking>_<model>` (`attention_zero_shot_*`, `attention_cssi_*`, `attention_residualized_ols_*`, `ig_lora_scgpt`, `attention_lora_geneformer`); `pipeline/postprocess/gene_sets.py` maps every method to its files and to its (ranking, model, model form). So one experiment (ranking, selection, model, dataset, cell type) is one row of `enrichment_summary/[<selection>/]summary_table.csv` in `results/dataset_<x>/`.

`results/legacy_v0/` holds the results of the earlier pipeline (old Dataset A balanced to 5261 cells per type, train + val only, old LoRA checkpoint, attention means over the cells expressing each gene); they are kept for reference and are not comparable with the current results.
