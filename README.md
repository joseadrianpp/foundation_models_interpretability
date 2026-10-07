# Foundation Models Interpretability

Run `run_tasks.py` for running the interpretability tasks, the same three for each model (scGPT with every gene, Geneformer V2 104M and scCello):

- **attention_scgpt**: forward-passes cells through the zero-shot pretrained scGPT model and captures the CLS-token attention to each gene. Genes are scored by `mean_attention_pd - mean_attention_control`, so the sign shows which condition attends to that gene more. Each mean is over every cell of the condition, a cell without the gene counting as 0, so a gene seen in only a few cells cannot rank above one seen in all of them.
- **attention_lora_scgpt**: the same attention score with the LoRA fine-tuned scGPT classifier, using only the cells it classifies correctly.
- **ig_lora_scgpt**: runs Integrated Gradients (Captum library) on the LoRA fine-tuned scGPT classifier, scoring each gene by its mean absolute attribution to the PD-classification logit over every correctly classified cell (0 where the gene is absent).
- **attention_geneformer**, **attention_lora_geneformer**, **ig_lora_geneformer**: the same three with Geneformer. Its cells are ranked gene tokens with no expression value, so Integrated Gradients runs over the token embeddings, from a baseline where every gene is the `<mask>` token, and a gene's attribution is the sum over the embedding dimensions.
- **attention_sccello**, **attention_lora_sccello**, **ig_lora_sccello**: the same three with scCello, a smaller Geneformer-like model that reads the same kind of cells.

Run `--task <name>` to run only one of them. `--task all` (default) runs the nine. `--dataset A` (default) or `--dataset B` picks the dataset; results go to `results/dataset_a/` or `results/dataset_b/`. `results/dataset_*/ig_lora_*/` already holds the Integrated Gradients top-30 of the three models in both datasets, computed with the same score, so those three tasks only need running to reproduce them.

Everything runs in the scGPT environment: Geneformer and scCello are plain BERTs loaded with `transformers`/`peft`, and their cells come already tokenized, so neither the `geneformer` nor the `sccello` package is needed.

## Data

`data/` is not in git; it comes in `fm_interpretability_data.tar.gz`. There are two datasets, Dataset A (`data/dataset_a/`) and Dataset B (`data/dataset_b/`), with the same four cell types. In each one the three models see the same cells and the same train / validation / test split (cell level, 80/10/10), and every cell has a `cell_type` and a binary `diagnosis`.

| Folder | Content |
|---|---|
| `dataset_*/mixed_split_scgpt/` | scGPT cells: normalised + log1p expression of every protein-coding gene in scGPT's vocabulary (`*_train/val/test.h5ad`, genes named as in the vocabulary), and `gene_names.csv` (token -> current gene symbol). scGPT reads up to 1200 expressed genes per cell |
| `dataset_*/mixed_split_geneformer/` | Geneformer cells, already tokenized (`*_train/val/test.dataset`), and `gene_names.csv` (token -> gene) |
| `dataset_*/mixed_split_sccello/` | scCello cells, already tokenized (`*_train/val/test.dataset`), and `gene_names.csv` (token -> gene) |
| `dataset_*/step5_1_lora_scgpt/` | LoRA fine-tuned scGPT classifier of that dataset (adapters merged into the model) |
| `dataset_*/step5_1_lora_geneformer/` | LoRA fine-tuned Geneformer classifier of that dataset (adapter + head, on top of `Geneformer_V2_104M/`) |
| `dataset_*/step5_1_lora_sccello/` | LoRA fine-tuned scCello classifier of that dataset (adapter + head, on top of `scCello_zeroshot/`) |
| `scGPT_brain/`, `Geneformer_V2_104M/`, `scCello_zeroshot/` | zero-shot pretrained models (scCello stored as a plain BERT) |
| `pd_panels/` | PanelApp Parkinson Disease and Complex Parkinsonism panel and ParkinsonsUK-UCL GO annotations (QuickGO) |
| `go3_refs/` | GO ontology and human GO annotations |

## Post-hoc gene-set analyses

These read the top-30 CSVs produced above; run them individually after the model tasks, in this order, with the same `--dataset`:

| Task | What it does | Output (inside `results/dataset_*/`) |
|---|---|---|
| `enrichment` | g:Profiler GO/KEGG/REAC enrichment | `enrichment/` |
| `enrichment_summary` | term-count/IC/functional abundance stats + pairwise Mann-Whitney tests over the enrichment above | `enrichment_summary/` |
| `gwas_similarity` | GO3 gene-level similarity (lin/SimRel/wang or any other distance) vs GWAS Catalog PD/AD/CAD gene sets and the PanelApp and ParkinsonsUK-UCL PD panels | `gwas_similarity/` |
| `gene_matrix` | gene x gene GO3 similarity matrices vs the Parkinson references (GWAS, PanelApp, ParkinsonsUK-UCL) | `gene_matrix/` |

```
python run_tasks.py --task <name> --dataset <A|B>
```
