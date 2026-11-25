# PD vs control classification of astrocytes from frozen embeddings

This folder is separate from the interpretability code of the rest of the repo and shares nothing with it. It answers one question: can a small MLP tell the astrocytes of Parkinson's disease (PD) donors from those of control donors using only the zero-shot cell embedding of a foundation model, for donors it has never seen? Each model (scGPT, Geneformer V2 104M and scCello) runs with its authors' own code, and the MLP head is the one of pipeline_unified's Step 4, the head the LoRA fine-tuning starts from.

```
python classification/embed_scgpt.py --dataset A        # scgpt_env
python classification/embed_geneformer.py --dataset A   # geneformer_env
python classification/embed_sccello.py --dataset A      # sccello_env
python classification/train_mlp.py --dataset A --model scgpt   # scgpt_env, once per model
python classification/plot_results.py [--umap]                 # scgpt_env, after every MLP
```

Run everything from the root of the repo, once with `--dataset A` and once with `--dataset B`. [`presenting_results.md`](presenting_results.md) explains how to read the results and how to present them.

## Data

`classification/data/` is not in git. It comes in `fm_classification_astrocytes_<date>.tar.gz`: unpack it at the root of the repo (`tar -xzf fm_classification_astrocytes_<date>.tar.gz`).

| Folder | Content |
|---|---|
| `dataset_a/`, `dataset_b/` | `astrocytes_{train,val,test}.h5ad`: the astrocytes of Dataset A and Dataset B (the same two datasets as the interpretability), raw counts |
| `scGPT_brain/`, `Geneformer-V2-104M/`, `scCello-zeroshot/` | the zero-shot pretrained models, as their authors published them |
| `githubs/Geneformer/`, `githubs/scCello/` | the authors' code, at the commits in `githubs/commits.json` |

The split is by donor: each donor has all its cells in one of the three files, so the test measures how the classifier does on people it has never seen. Validation and test have 2 PD and 2 control donors each, the same donors as the donor-split runs of the paper, chosen so that the three sets have the same mix of the four cell types.

| | Train | Validation | Test |
|---|---|---|---|
| Dataset A, donors PD / control | 11 / 10 | 2 / 2 | 2 / 2 |
| Dataset A, cells PD / control | 8060 / 7791 | 1468 / 2503 | 2942 / 1487 |
| Dataset B, donors PD / control | 19 / 5 | 2 / 2 | 2 / 2 |
| Dataset B, cells PD / control | 22 283 / 2007 | 2733 / 447 | 2577 / 3096 |

Each h5ad has, per cell (`obs`): `donor`, `diagnosis` (`PD` or `control`), `diagnosis_raw` (as the dataset spells it; Dataset B splits PD into PD and PD with dementia), `cell_type` and the donor's `age`, `sex`, `pmi` and `rin` (Dataset A) or `braak_stage` (Dataset B). The genes are the protein-coding ones outside the sex chromosomes and the mitochondrial genome (18 318 in A, 17 119 in B), named by their current symbol. Each model reads the genes its vocabulary knows, by the name in its own `var` column (empty if it does not read the gene):

| Column | Model | Genes A / B |
|---|---|---|
| `scgpt_gene` | scGPT, by symbol (the vocabulary still uses some older symbols, e.g. GBA for GBA1) | 18 297 / 17 101 |
| `ensembl_id_geneformer` | Geneformer, by Ensembl ID through its own mapping dictionary | 18 308 / 17 110 |
| `ensembl_id_sccello` | scCello, by Ensembl ID (HGNC's, else scCello's own table) | 18 186 / 17 016 |

## Environments

Each model needs its own environment, with the versions used for the paper:

| Environment | Scripts | How to build it |
|---|---|---|
| `scgpt_env` | `embed_scgpt.py`, `train_mlp.py`, `plot_results.py` | the repo's `requirements.txt`, plus `flash-attn==1.0.4` (the same one as the interpretability) |
| `geneformer_env` | `embed_geneformer.py` | see below |
| `sccello_env` | `embed_sccello.py` | see below |

```
python3.10 -m venv geneformer_env && source geneformer_env/bin/activate
pip install --no-deps -r classification/envs/geneformer_env.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install --no-deps -e classification/data/githubs/Geneformer

python3.10 -m venv sccello_env && source sccello_env/bin/activate
pip install --no-deps -r classification/envs/sccello_env.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

scCello's code is not installed: `embed_sccello.py` imports it from `data/githubs/scCello`, as the authors' own scripts do.

## What each script does

| Script | What it does |
|---|---|
| `embed_scgpt.py` | genes of the vocabulary, normalisation to 10^4 counts + log1p, and `scgpt.tasks.embed_data` (51 expression bins, up to 1200 genes per cell, `<cls>` of the last layer, L2-normalised), as in scGPT's zero-shot tutorials |
| `embed_geneformer.py` | Geneformer's `TranscriptomeTokenizer` (rank value encoding, up to 4096 tokens) and `EmbExtractor` (`<cls>` of the second-to-last layer) |
| `embed_sccello.py` | scCello's `ProcessSingleCellData` and `TranscriptomeTokenizer` (rank value encoding, up to 2048 tokens) and the encoder of its classification model (`<cls>` of the last layer) |
| `train_mlp.py` | MLP head on one model's embeddings: train balanced 50/50, early stopping on the validation balanced accuracy, test per cell and per donor |
| `plot_results.py` | the summary table and the figures of every finished MLP |

Geneformer's and scCello's embeddings are those of the paper's runs (largest difference per value 5e-3 and 8e-6). scGPT's are not exactly: `embed_data` gives the `<cls>` token the value -2 and L2-normalises the output, while the paper's runs gave it 0 and kept the raw output, as scGPT's fine-tuning tutorials do. The two embeddings of a cell have a median cosine similarity of 0.986.

## Results

Everything goes to `classification/results/` (not in git):

```
results/dataset_<a|b>/<model>/embeddings.h5ad        one row per cell, with its obs
results/dataset_<a|b>/<model>/mlp/seed_<seed>/
    best_mlp.pt            the head (same format as pipeline_unified's, loads into FlexMLP)
    training_history.csv   loss and validation balanced accuracy per epoch; `kept` marks the epoch kept
    predictions.csv        PD logit and probability of every cell (train, val and test)
    test_metrics.json      test (and validation) metrics per cell, and test donors classified correctly
    test_donors.csv        mean PD probability of each test donor's cells and the call it gives
results/figures/           summary.csv and the figures (PNG and PDF)
```

`train_mlp.py --seed <n>` trains the same head with another seed in its own `seed_<n>/` folder; with several seeds, `plot_results.py` reports the mean over them and its 95% CI.
