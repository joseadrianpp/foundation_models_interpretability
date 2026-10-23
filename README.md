# Foundation Models Interpretability

Run `run_tasks.py` for running the interpretability tasks of scGPT (with every gene), Geneformer V2 104M and scCello, for Parkinson's disease (PD) vs control. Each model comes in four forms:

- **Zero-shot** (`attention_<model>`): the pretrained model. Forward-passes every cell and captures the CLS-token attention to each gene. Genes are scored by `mean_attention_pd - mean_attention_control`, so the sign shows which condition attends to that gene more. Each mean is over the cells of the condition that have the gene, so it measures the gene where it is expressed and not how many cells express it, and a gene is scored only if at least 5 % of the cells of each condition have it (`MIN_PRESENT_FRAC` in `pipeline/config.py`). Every other mean (CSSI strata, donors, Integrated Gradients) follows the same rule.
- **Step 2.5** (`attention_step2_5_<model>`): the pretrained model after a further self-supervised pretraining on the training cells of that dataset, with the model's own pretraining objective and no diagnosis labels. It has no head, so it runs the zero-shot attention task.
- **Zero-shot + MLP head** (`attention_mlp_<model>`, `ig_mlp_<model>`): the zero-shot model, not fine-tuned, with the small MLP head trained to classify PD from its frozen cell embedding (the head the LoRA training starts from). The attention score uses only the cells it classifies correctly, and Integrated Gradients (Captum) scores each gene by its mean absolute attribution to the PD logit over those of them that have the gene (at least 5 % of them).
- **LoRA** (`attention_lora_<model>`, `ig_lora_<model>`): the same two tasks with the LoRA fine-tuned classifier.

scGPT's Integrated Gradients run over the binned expression values from an all-zero baseline. Geneformer and scCello cells are ranked gene tokens with no expression value, so Integrated Gradients run over the token embeddings, from a baseline where every gene is the `<mask>` token, and a gene's attribution is the sum over the embedding dimensions.

Every score is then ranked three ways and every ranking cut three ways:

| | |
|---|---|
| Ranking | **pooled** (the score above); **CSSI**: Leiden strata of the model's cell embedding, the same score inside each stratum with enough cells per condition, and each gene's maximum; **residualized**: the OLS residual of the pooled score against each gene's mean expression, variance and dropout rate (`<score>_residualized_ols_<form>_<model>` tasks, CPU) |
| Selection | **top30**; **z-score** (score > mean + 2.5 sd); **elbow** (Kneedle) |

Run `--task <name>` to run only one of them. `--task all` (default) runs every model task and then every OLS task. `--dataset A` (default) or `--dataset B` picks the dataset; results go to `results/dataset_a/` or `results/dataset_b/`. `--max-cells-per-type N` runs a seeded subset of N cells per cell type (quick checks; the OLS tasks need the same N), and `--skip-done` keeps the cell types a previous run already finished (`done.json`), so a run cut short continues where it stopped.

Everything runs in the scGPT environment (`requirements.txt`, plus `flash-attn==1.0.4` for scGPT): Geneformer and scCello are plain BERTs loaded with `transformers`/`peft`, and their cells come already tokenized, so neither the `geneformer` nor the `sccello` package is needed.

## Data

`data/` is not in git; it comes in `fm_interpretability_data.tar.gz`. There are two datasets, Dataset A (`data/dataset_a/`) and Dataset B (`data/dataset_b/`), with the same four cell types. In each one the three models see the same cells and the same train / validation / test split (cell level, 80/10/10), and every cell has a `cell_type` and a binary `diagnosis`. The tasks use train + validation + test together.

| Folder | Content |
|---|---|
| `dataset_*/mixed_split_scgpt/` | scGPT cells: normalised + log1p expression of every protein-coding gene in scGPT's vocabulary (`*_train/val/test.h5ad`, genes named as in the vocabulary, with a `donor` column), and `gene_names.csv` (token -> current gene symbol). scGPT reads up to 1200 expressed genes per cell |
| `dataset_*/mixed_split_geneformer/` | Geneformer cells, already tokenized (`*_train/val/test.dataset`), and `gene_names.csv` (token -> gene) |
| `dataset_*/mixed_split_sccello/` | scCello cells, already tokenized (`*_train/val/test.dataset`), and `gene_names.csv` (token -> gene) |
| `dataset_*/step5_1_lora_scgpt/` | LoRA fine-tuned scGPT classifier of that dataset (adapters merged into the model) |
| `dataset_*/step5_1_lora_geneformer/` | LoRA fine-tuned Geneformer classifier of that dataset (adapter + head, on top of `Geneformer_V2_104M/`) |
| `dataset_*/step5_1_lora_sccello/` | LoRA fine-tuned scCello classifier of that dataset (adapter + head, on top of `scCello_zeroshot/`) |
| `dataset_*/step4_mlp_{scgpt,geneformer,sccello}/` | MLP head of the zero-shot + MLP classifier of that dataset (`best_mlp.pt`, on top of the zero-shot model) |
| `dataset_*/step2_5_scgpt/` | scGPT after Step 2.5 on that dataset (`best_model.pt`, same architecture and vocabulary as `scGPT_brain/`) |
| `dataset_*/step2_5_geneformer/`, `dataset_*/step2_5_sccello/` | Geneformer and scCello after Step 2.5 on that dataset (scCello stored as a plain BERT, like `scCello_zeroshot/`) |
| `scGPT_brain/`, `Geneformer_V2_104M/`, `scCello_zeroshot/` | zero-shot pretrained models (scCello stored as a plain BERT) |
| `pd_panels/` | PanelApp Parkinson Disease and Complex Parkinsonism panel and ParkinsonsUK-UCL GO annotations (QuickGO) |
| `go3_refs/` | GO ontology and human GO annotations |

## Post-hoc gene-set analyses

These read the gene sets produced above; run them after the model and OLS tasks, in this order (or all at once with `--task postprocess`), with the same `--dataset`. `--selection` picks the gene sets (default all three). `enrichment` and `gwas_similarity` need internet access.

| Task | What it does | Output (inside `results/dataset_*/`) |
|---|---|---|
| `enrichment` | g:Profiler GO/KEGG/REAC enrichment | `enrichment/` |
| `enrichment_summary` | term-count/IC/functional abundance stats + pairwise Mann-Whitney tests over the enrichment above | `enrichment_summary/` |
| `gwas_similarity` | GO3 gene-level similarity (lin/SimRel/wang or any other distance) vs GWAS Catalog PD/AD/CAD/kidney gene sets and the PanelApp and ParkinsonsUK-UCL PD panels | `gwas_similarity/` |
| `gene_matrix` | gene x gene GO3 similarity matrices vs the Parkinson references (GWAS, PanelApp, ParkinsonsUK-UCL) | `gene_matrix/` |
| `plots` | the comparison figures of the tasks above, one per model | `plots/` |

`python -m pipeline.macrotable.build` then joins every experiment (ranking x score x selection x model x form x dataset x cell type) into `results/macrotable/macrotable.csv`, with `SimPD` = similarity to the three PD references minus that to the three control diseases. `evaluate_attributions.py` scores per-cell attributions with Quantus (deletion curves against random attributions). `scripts/check_run.py` checks a finished run.

```
python run_tasks.py --task <name> --dataset <A|B>
```
