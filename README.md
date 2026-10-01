# Foundation Models Interpretability

Run `run_tasks.py` for running the interpretability tasks:

- **attention_scgpt**: forward-passes cells through the zero-shot pretrained scGPT model and captures the CLS-token attention to each gene. Genes are scored by `mean_attention_pd - mean_attention_control`, so the sign shows which condition attends to that gene more. Each mean is over every cell of the condition, a cell without the gene counting as 0, so a gene seen in only a few cells cannot rank above one seen in all of them.
- **ig_lora_scgpt**: runs Integrated Gradients (Captum library) on the LoRA fine-tuned scGPT classifier, scoring each gene by its mean absolute attribution to the PD-classification logit over every correctly classified cell (0 where the gene is absent).
- **attention_geneformer**: the same attention score as `attention_scgpt`, with the zero-shot pretrained Geneformer (V2, 104M) model.
- **attention_lora_geneformer**: the same attention score with the LoRA fine-tuned Geneformer classifier, using only the cells it classifies correctly.

Run `--task <name>` to run only one of them. `--task all` (default) runs the four. `--dataset A` (default) or `--dataset B` picks the dataset; results go to `results/dataset_a/` or `results/dataset_b/`.

Everything runs in the scGPT environment: Geneformer is a plain BERT loaded with `transformers`/`peft`, and its cells come already tokenized, so the `geneformer` package is not needed.

## Data

`data/` is not in git; it comes in `fm_interpretability_data.tar.gz`. There are two datasets, Dataset A (`data/dataset_a/`) and Dataset B (`data/dataset_b/`), with the same four cell types. In each one both models see the same cells and the same train / validation / test split (cell level, 80/10/10), and every cell has a `cell_type` and a binary `diagnosis`.

| Folder | Content |
|---|---|
| `dataset_*/mixed_split_scgpt/` | scGPT cells: normalised + log1p expression of 1000 HVG (`*_train/val/test.h5ad`) |
| `dataset_*/mixed_split_geneformer/` | Geneformer cells, already tokenized (`*_train/val/test.dataset`), and `gene_names.csv` (token -> gene) |
| `dataset_*/step5_1_lora_scgpt/` | LoRA fine-tuned scGPT classifier of that dataset (adapters merged into the model) |
| `dataset_*/step5_1_lora_geneformer/` | LoRA fine-tuned Geneformer classifier of that dataset (adapter + head, on top of `Geneformer_V2_104M/`) |
| `scGPT_brain/`, `Geneformer_V2_104M/` | zero-shot pretrained models |
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
