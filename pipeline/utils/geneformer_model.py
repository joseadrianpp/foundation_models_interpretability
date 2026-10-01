"""
Geneformer V2 (104M) vocabulary, gene names and LoRA classifier.

Geneformer is a plain BERT, so the backbone is loaded with transformers and
the LoRA adapter with peft (see steps/_step7_geneformer.py). The head on top
is the same FlexMLP class as the scGPT one (utils/scgpt_model.py).
"""

import pickle
from pathlib import Path

import pandas as pd
from torch import nn

from .. import config as C

# The classification head reads the CLS token after the second-to-last encoder
# block, which is the cell embedding Geneformer's own EmbExtractor uses
EMB_HIDDEN_LAYER = -2


# Loading the Geneformer vocabulary (Ensembl ID -> token id)
def load_token_dictionary() -> dict:
    with open(C.GENEFORMER_TOKEN_DICT, "rb") as f:
        return pickle.load(f)


# Loading the token id -> gene symbol map of the genes in one dataset
def load_gene_names(data_dir: Path) -> dict:
    df = pd.read_csv(data_dir / C.GENEFORMER_GENE_NAMES)
    return dict(zip(df["token_id"], df["gene"]))


# Backbone + FlexMLP head, the same classifier the LoRA was trained as
class GeneformerClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, attention_mask):
        out = self.backbone.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)
        # CLS token (position 0) of the second-to-last block, one PD logit per cell
        return self.head(out.hidden_states[EMB_HIDDEN_LAYER][:, 0, :]).squeeze(1)
