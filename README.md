# Foundation Models Interpretability

Gene-level interpretability of three single-cell foundation models, **scGPT** (every gene), **Geneformer (V2, 104M)** and **scCello**, for Parkinson's disease (PD) vs control, on two datasets and four cell types (astrocytes, DA neurons, microglia, oligodendrocytes).

For every (model, dataset, cell type) the pipeline scores genes three ways (zero-shot attention, LoRA attention, LoRA Integrated Gradients), ranks each score with three methods, cuts each ranking into a gene set with three selection rules, and evaluates every gene set with GO enrichment and GO semantic similarity to PD reference panels:

| Dimension | Values |
|---|---|
| Ranking | pooled score (attention: PD - control; IG: mean \|IG\|), CSSI-max, OLS-residualized |
| Score | attention (zero-shot or LoRA), Integrated Gradients (LoRA) |
| Selection (top-N) | top-30, z-score, elbow |
| Model | scGPT, Geneformer, scCello (zero-shot and LoRA fine-tuned) |
| Dataset | A, B |
| Cell type | astrocytes, DA neurons, microglia, oligodendrocytes |

## Environment

```
python -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

Everything runs in this one environment: Geneformer and scCello are plain BERTs loaded with `transformers`/`peft` and their cells come already tokenized, so neither the `geneformer` nor the `sccello` package is needed. `go3` (GO semantic similarity, Rust) installs from PyPI as a prebuilt wheel. scGPT's flash-attention backend needs a CUDA GPU for the model tasks; the OLS and post-hoc tasks run on CPU.

## Data

`data/` is not in git. Expected layout:

```
data/
├── scGPT_brain/                 scGPT zero-shot pretrained model (args.json, vocab.json, best_model.pt)
├── Geneformer_V2_104M/          Geneformer zero-shot pretrained model + token_dictionary_gc104M.pkl
├── scCello_zeroshot/            scCello zero-shot pretrained model, stored as a plain BERT, + token_dictionary.pkl
├── go3_refs/                    GO ontology (go-basic.obo) and human GO annotations (goa_human.gaf)
├── pd_panels/                   PanelApp "Parkinson Disease and Complex Parkinsonism" panel and
│                                ParkinsonsUK-UCL GO annotations (QuickGO)
├── dataset_a/
│   ├── mixed_split_scgpt/       scGPT cells: normalised + log1p expression of every protein-coding gene
│   │                            of scGPT's vocabulary (Mixed_allgenes_*_train/val/test.h5ad), and
│   │                            gene_names.csv (token -> current gene symbol, e.g. GBA -> GBA1)
│   ├── mixed_split_geneformer/  Geneformer cells, already tokenized (*_train/val/test.dataset),
│   │                            and gene_names.csv (token -> gene symbol)
│   ├── mixed_split_sccello/     scCello cells, the same way
│   ├── step5_1_lora_scgpt/      LoRA fine-tuned scGPT classifier (adapters merged, best_full_model.pt)
│   ├── step5_1_lora_geneformer/ LoRA fine-tuned Geneformer classifier (best_lora/ + best_head.pt)
│   └── step5_1_lora_sccello/    LoRA fine-tuned scCello classifier (best_lora/ + best_head.pt)
└── dataset_b/                   same layout as dataset_a
```

In each dataset the three models see the same cells, in the same order, and the same train / validation / test split (cell level, 80/10/10). Every cell has a `cell_type` and a binary `diagnosis`; the scGPT h5ad files also have a `donor` column (used only by `--permutation`). The pipeline always uses train + val + test together.

## Running

```
python run_tasks.py --task <task> --dataset <A|B|A+B> [options]
```

Each dataset writes to its own folder, `results/dataset_a/` or `results/dataset_b/`, so the post-hoc analyses never mix them.

### Model tasks (GPU)

Three per model (`<model>` = `scgpt`, `geneformer`, `sccello`):

| Task | What it does | Output folder |
|---|---|---|
| `attention_<model>` | zero-shot model: pooled attention, CSSI and their selections | `attention_zero_shot_<model>/` |
| `attention_lora_<model>` | attention of the LoRA classifier (correctly classified cells): pooled, CSSI and their selections | `attention_lora_<model>/` |
| `ig_lora_<model>` | Integrated Gradients on the LoRA classifier (correctly classified cells): pooled, CSSI and their selections | `ig_lora_<model>/` |

### OLS tasks (CPU, after the model tasks)

One per model task, each in the folder of its name:

| Task | What it does |
|---|---|
| `attention_residualized_ols_<model>` | OLS residual of the zero-shot pooled attention and its selections |
| `attention_residualized_ols_lora_<model>` | the same for the LoRA attention |
| `ig_residualized_ols_lora_<model>` | the same for the LoRA IG score |

### Post-hoc tasks (CPU, after the tasks above)

Run in this order, or all at once with `--task postprocess`. `enrichment` (g:Profiler) and `gwas_similarity` (GWAS Catalog, cached after the first run) need internet access.

| Task | What it does | Output folder |
|---|---|---|
| `enrichment` | g:Profiler GO:BP/MF/CC, KEGG and Reactome enrichment of every gene set | `enrichment/` |
| `enrichment_summary` | term count, information content, intersection size and functional abundance per gene set, plus pairwise Mann-Whitney tests between the methods of each model (BH-adjusted) | `enrichment_summary/` |
| `gwas_similarity` | GO3 set-level similarity (lin/SimRel/wang, BMA) of every gene set to the PD/AD/CAD/CKD GWAS Catalog gene sets and the two PD panels | `gwas_similarity/` |
| `gene_matrix` | gene x gene GO3 similarity matrices vs the three PD references (GWAS, PanelApp, ParkinsonsUK-UCL) | `gene_matrix/` |
| `plots` | every comparison figure, one per model (also drawn by the tasks above) | `plots/` |

### Options

| Option | Meaning |
|---|---|
| `--task all` (default) | the nine model tasks, then the nine OLS tasks |
| `--dataset A\|B\|A+B` | dataset (default A). `A+B` pools the cells of both (A first) into `results/dataset_a+b/`; only the Geneformer and scCello zero-shot tasks can run on it, since the scGPT h5ad files of A and B keep different genes and the LoRA models are trained per dataset |
| `--gpus 0,1` | model tasks only: one worker per GPU, whole cell types balanced by cell count |
| `--device cuda:0` | GPU for single-GPU runs |
| `--selection top30\|z-score\|elbow\|all` | gene-set selections used by the post-hoc tasks (default all) |
| `--permutation` | also run the donor-level permutation selector (scGPT only, slow) |
| `--donor-col` | donor column for `--permutation` (auto-detected, `donor` in both datasets) |
| `--max-cells-per-type N` | seeded random subset of at most N cells per cell type, for quick test runs; the OLS tasks must use the same N as the model tasks |
| `--skip-done` | model tasks keep every cell type a previous run of the same task finished with the same inputs and settings (`done.json` in its folder), so a run cut short continues where it stopped |
| `--batch-scale auto\|N` | multiplier of the model batch sizes, which are tuned for an 8 GB GPU; `auto` (default) adds x1 per 10 GiB of GPU memory, up to x8 (RTX 2070 x1, L4 x2, L40S x4, H100 x7, H200 x8) |
| `--distances lin SimRel wang` | GO term similarities computed by `gwas_similarity` / `gene_matrix` (default all three; the plots use SimRel) |
| `--results-dir` | results root (default `results/`) |

Set `GO3_THREADS` to limit the CPU threads of the GO3 tasks (default: every core). Run with `python -u` (or `PYTHONUNBUFFERED=1`) when the output goes to a log file, so progress shows up as it happens.

Integrated Gradients over every correctly classified cell takes hours on the large cell types (Geneformer most), so every IG task saves the attributions computed so far to `ig_checkpoint.npz` in the cell type's folder every 15 minutes, and a rerun of the task resumes from it (with or without `--skip-done`); the file is deleted once the cell type is written. The CSSI strata of a LoRA model are computed once: its attention and IG tasks compute the same cell embeddings, and whichever runs second reuses the other's Leiden strata.

The long loops (model forward passes, IG, Leiden resolutions, GO3 gene pairs, g:Profiler queries, gene matrices) show tqdm progress bars. When the output goes to a file, or with `--gpus` (several workers share one output), each bar is printed as one line every 30 s (`PROGRESS_INTERVAL` changes it) instead of being redrawn, so the logs stay readable; follow one task with e.g. `tail -f logs/B_attention_geneformer.log`.

### Run time

The GO3 similarity of `gwas_similarity` and `gene_matrix` dominates the post-hoc time: it grows with (genes in the gene sets) x (genes in the references, 1646 in total, 799 of them in the ParkinsonsUK-UCL panel), and the Geneformer z-score/elbow sets hold hundreds of genes. Every gene-to-gene similarity is therefore computed once per (ontology, distance) and cached in `results/go3_pair_cache/` (shared by both datasets, every selection and both tasks; it only depends on the GO files, delete it if `data/go3_refs/` changes). For the gene sets of a 200-cells-per-type run, the first post-hoc run is estimated at ~2.3 h on 24 cores with three distances and ~0.8 h with `--distances SimRel` (~9-11 h before the cache); later runs only compute pairs of genes not seen before.

### Full run

```
for D in A B; do
  for M in scgpt geneformer sccello; do
    for T in attention_$M attention_lora_$M ig_lora_$M; do
      python -u run_tasks.py --task $T --dataset $D --gpus 0,1
    done
  done
  for M in scgpt geneformer sccello; do
    for T in attention_residualized_ols_$M attention_residualized_ols_lora_$M ig_residualized_ols_lora_$M; do
      python -u run_tasks.py --task $T --dataset $D
    done
  done
  python -u run_tasks.py --task postprocess --dataset $D
done
```

`scripts/run_full.sh` runs all of this unattended (the post-hoc tasks of A overlap with the model tasks of B), using every visible GPU (at most 4) and writing one log per task to `logs/`; `DATASETS`, `STAGES` (`model ols post`) and `MODEL_TASKS` run a part of it, e.g. only the model tasks on a GPU machine and the rest on a CPU one:

```
nohup bash scripts/run_full.sh > /dev/null 2>&1 &
tail -f logs/run_full.log
STAGES=model SKIP_DONE=1 bash scripts/run_full.sh             # GPU machine: the nine model tasks
STAGES=model MODEL_TASKS="ig_lora_geneformer" DATASETS=B bash scripts/run_full.sh
STAGES="ols post" bash scripts/run_full.sh                    # CPU machine, with the model outputs in results/
```

`PYTHON`, `GPUS`, `DATASETS`, `STAGES`, `MODEL_TASKS`, `SKIP_DONE`, `DISTANCES`, `GO3_THREADS`, `LOG_DIR` and `PROGRESS_INTERVAL` override its defaults (see the script header). `watch -n 30 bash scripts/progress.sh` shows the latest progress line of every running bar. When the run ends, `python scripts/check_run.py` checks logs, cell counts, CSSI strata, every experiment's gene list and every post-hoc output.

A quick end-to-end check before a full run: `--task all --max-cells-per-type 200`, then `--task postprocess --selection top30 --distances SimRel` (~20 min on 24 cores).

Measured on one L4 (24 GB) with the four tasks of the scGPT (1000 HVG) and Geneformer zero-shot / LoRA attention version: model tasks ~6 h for A and ~12 h for B (B's oligodendrocytes, 172k cells, the longest), post-hoc ~5 h for A and ~1.5 h for B with three distances. Integrated Gradients is much longer than an attention pass (50 steps, each a forward and a backward pass); measured on an L4 and extrapolated to every correctly classified cell: Geneformer ~415 tokens/s (~90 h for A, ~115 h for B), scCello ~3100 tokens/s (~10 h and ~15 h), scGPT ~0.5 s per cell (~12 h and ~30 h); an L40S is about three times faster. Larger GPUs shorten the model tasks; the post-hoc time depends on the CPU cores.

### Experiment table

`python -m pipeline.macrotable.build` joins every experiment (ranking x score x selection x model x model form x dataset x cell type) into `results/macrotable/macrotable.csv`, one row per gene list (key columns `Ranking` attention/CSSI/residuals/IG, `Score` attention/IG, `TopN`, `Model`, `ModelForm`, `Dataset`, `CellType`; the gene list, `genes`, is the last column; `SimPD` = SimPDGWAS + SimPDGE + SimPDGO - SimAD - SimKidney - SimCAD), from the outputs already computed (no inference). Each group of columns comes from one module of `pipeline/macrotable/columns/` and is saved to `results/macrotable/parts/<module>.csv`; `--modules <module>` recomputes only those parts and joins the table again, and the random forest results are cached per gene list in `parts/classifier_cache.csv`.

### Moving to another server

1. Clone the repository and copy `data/` (~2.5 GB, not in git), e.g. `rsync -a old-server:path/to/repo/data/ data/`. `results/go3_pair_cache/` can be copied too, to reuse GO3 similarities already computed.
2. Python 3.10 environment. Install torch from the PyTorch wheel index first, then the rest:
   ```
   pip install torch==2.3.0 torchtext==0.18.0 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
3. flash-attention 1.x (scGPT's attention backend) compiles from source against the local CUDA toolkit: `pip install flash-attn==1.0.4 --no-build-isolation`, with `nvcc` from a CUDA 12.x toolkit on the PATH (torch is built for CUDA 12.1). Ada (L40) runs Ampere kernels. On Hopper (H100, H200) the build must include sm_90 kernels; if scGPT fails with a "no kernel image" CUDA error, rebuild it with sm_90 added to the architectures in flash-attn's `setup.py`. Geneformer uses PyTorch SDPA and does not need it.
4. Check: `python -c "import flash_attn, go3, scgpt, transformers, peft"`, then the quick end-to-end check above on dataset A, then `scripts/run_full.sh`.

`visualize_cssi_embeddings.py --dataset <A|B|A+B> [--scores ...]` draws, per score with CSSI (zero-shot attention, LoRA attention and LoRA IG of every model) and cell type, the cell embeddings and Leiden strata CSSI used, colored by stratum and by condition, with each stratum's cells per condition (no inference; `plots/cssi_embeddings/<scores>/<ct>.png`).

## Methods

### Gene rankings

All attention scores use the CLS-token attention of one encoder layer (scGPT layer 11, Geneformer layer 10 of 12, scCello layer 5 of 6: the block whose output is the cell embedding the classifiers read), computed from Q·Kᵀ of the CLS row, rank-normalised over each cell's genes and averaged over heads. One forward pass per cell type feeds every ranking. scGPT reads up to 1200 expressed genes per cell (a cell with more keeps a random 1200, seeded like the binning); genes are named by their current symbol (`gene_names.csv`).

- **Pooled attention** (`attention_zero_shot_<model>`): `mean_attention_pd - mean_attention_control` per gene. Each mean is over every cell of the condition, a cell without the gene counting as 0, so a gene seen in only a few cells cannot rank above one seen in all of them. The sign shows which condition attends to the gene more.
- **CSSI-max** (`attention_cssi_<model>`): the cells are split into ~6 strata by Leiden clustering (cosine k-NN graph, k = 15) of the residual-stream CLS embedding entering the attention layer, at the resolution (of 25 between 0.1 and 2.5, tried in increasing order, in batches of 8 parallel processes on cell types of 10k cells or more, and stopping after 3 in a row with more than 6 clusters; `resolution_sweep` in `cssi_metadata.json`) whose number of clusters is closest to 6; the same PD - control score is computed inside every stratum with at least 20 cells per condition, and each gene keeps its maximum. Without any valid stratum it falls back to the pooled score.
- **OLS-residualized** (`attention_residualized_ols_<model>`): across genes, the pooled score is regressed on three gene-level covariates of the same cells (mean expression, expression variance, dropout rate); the residual is the part of the attention not explained by how much / how often a gene is expressed.
  - scGPT: the covariates come from the h5ad expression matrix.
  - Geneformer and scCello: cells are rank-value encoded (no expression values), so the covariates use a rank pseudo-expression, `v = 1 - (p - 1) / (n - 1)` for the gene at position `p` of a cell's `n` genes (1 = the cell's most expressed gene) and `v = 0` when the gene is absent.

Fine-tuned (LoRA) models, two scores, each with the same three rankings, on their correctly classified cells:

- **LoRA attention** (`attention_lora_<model>`): the pooled attention score of the LoRA classifier, on its correctly classified cells.
- **Integrated Gradients** (`ig_lora_<model>`): mean absolute attribution to the PD logit of the LoRA classifier over every correctly classified cell (0 where the gene is absent), 50 Gauss-Legendre steps (Captum). scGPT: w.r.t. the binned expression values, from an all-zero baseline. Geneformer and scCello have no expression value (the expression level is the gene's position), so IG runs over the token embeddings, from a baseline where every gene is the `<mask>` token (the CLS and the positions are the same at both ends), and a gene's attribution is the sum over the embedding dimensions; in bf16 on the GPU.
- **CSSI-max** (`attention_cssi_lora_<model>`, `ig_cssi_lora_<model>`): Leiden strata (as above) of the LoRA model's own CLS embedding, over every cell; each gene keeps its maximum over the valid strata of the stratum's score: their PD - control attention, or mean |IG| over the stratum's correctly classified cells. A stratum is valid with at least 20 correctly classified cells per condition, so every mean rests on enough cells; the fine-tuned embedding groups cells by the model's prediction, so only 1-3 strata qualify on Dataset B. The per-cell scores are kept in `cell_scores.npz`, so CSSI can be recomputed on CPU (`AttentionCache.load`, `IGResult.load`).
- **OLS-residualized** (`attention_residualized_ols_lora_<model>`, `ig_residualized_ols_lora_<model>`): the same OLS on the pooled score, with the covariates of every cell of the cell type.

### Gene-set selections

Each ranking is cut three ways (files `top30.csv`, `z-score.csv`, `elbow.csv`, columns `rank, gene, score`):

- **top30**: the 30 highest scores.
- **z-score**: genes with score > mean + 2.5 sd of all the gene scores.
- **elbow**: Kneedle elbow of the descending score curve (falls back to top-30 if there is no elbow).

`selection_summary.json` records the size and parameters of each set. With `--permutation`, a donor-level permutation selector (label permutations of donors, Cohen's d, BH q < 0.05 and |d| ≥ effect size) is also written (`permutation.csv`, `permutation_audit.csv`); it needs donor labels, so it is scGPT only.

### Gene-set metrics

- **Enrichment** (g:Profiler, significant terms only): number of terms, information content `-log2(term_size / domain_size)`, intersection size, functional abundance `intersection_size / term_size`.
- **GO semantic similarity** (GO3, BP/CC/MF, lin/SimRel/wang): set-level BMA similarity to the PD GWAS gene set (GWAS Catalog), the PanelApp PD panel (green genes) and the ParkinsonsUK-UCL GO-annotated genes, with Alzheimer, coronary artery disease and chronic kidney disease GWAS sets as controls; gene x gene matrices vs the three PD references.

### Reproducibility

scGPT's binning breaks ties randomly and its padding keeps a random 1200 genes of a cell with more, so tokenisation is seeded (`BINNING_SEED = 42` in `pipeline/config.py`); Leiden, the subsets of `--max-cells-per-type` and the permutations use seed 42. Results are identical across runs and across single- and multi-GPU runs on the same GPU model and `--batch-scale`. A different GPU or batch size changes the attention scores only by floating-point rounding (relative differences ~1e-4, from fp16 and padding), which can swap genes with nearly tied scores.

## Results layout

Inside `results/dataset_<a|b|a+b>/` (`<ct>` = cell type, `<model>` = `scgpt` / `geneformer` / `sccello`):

```
attention_zero_shot_<model>/<ct>/
├── top30_genes.csv              pooled ranking, top-30
├── csv/all_genes.csv            pooled ranking, every gene
├── csv/{top30,z-score,elbow}.csv + selection_summary.json
├── cssi_genes.csv               CSSI ranking, every gene
├── cssi/csv/{top30,z-score,elbow}.csv + selection_summary.json
├── cell_embeddings.npy, leiden_clusters.csv, cssi_metadata.json
├── cell_scores.npz              LoRA folders only: per-cell token scores
├── done.json                    the task finished this cell type (inputs and settings, for --skip-done)
├── ig_checkpoint.npz            IG folders, only while the task is unfinished: IG computed so far
attention_residualized_ols_<model>/<ct>/
├── residualized_genes.csv       score, OLS fit, residual and covariates of every gene
├── csv/ols.csv                  residual ranking, every gene
├── csv/{top30,z-score,elbow}.csv + selection_summary.json
└── ols_summary.json             R², coefficients, F-test
attention_lora_<model>/<ct>/, ig_lora_<model>/<ct>/
                                 the same files as attention_zero_shot_<model>/<ct>/
attention_residualized_ols_lora_<model>/<ct>/, ig_residualized_ols_lora_<model>/<ct>/
                                 the same files as attention_residualized_ols_<model>/<ct>/
enrichment/[<selection>/]<method>/<ct>.csv, manifest.csv
enrichment_summary/[<selection>/]summary_table.csv, pairwise_tests.csv
gwas_similarity/refs/            reference gene sets (GWAS Catalog cached on first use)
gwas_similarity/[<selection>/]gene_level_similarity_all.csv
gene_matrix/[<selection>/]matrices/<ct>/<reference>__<distance>__<ontology>.csv, summary_statistics.csv
plots/<figure>_<model>[_<selection>].png
```

`results/go3_pair_cache/<ontology>__<distance>.npz` holds the cached gene-to-gene GO3 similarities shared by both datasets.

The top-30 outputs of the post-hoc tasks have no `<selection>/` subfolder. A method name in those outputs is `<ranking>_<model>` (`attention_zero_shot_*`, `attention_cssi_*`, `attention_residualized_ols_*`, and for the LoRA models `attention_lora_*`, `attention_cssi_lora_*`, `attention_residualized_ols_lora_*`, `ig_lora_*`, `ig_cssi_lora_*`, `ig_residualized_ols_lora_*`); `pipeline/postprocess/gene_sets.py` maps every method to its files and to its (ranking, score, model, model form). So one experiment (ranking, score, selection, model, dataset, cell type) is one row of `enrichment_summary/[<selection>/]summary_table.csv` in `results/dataset_<x>/`. In the figures (one per model) the color is the ranking and the texture the score (solid: zero-shot attention, `////`: LoRA attention, `xxxx`: LoRA IG).
