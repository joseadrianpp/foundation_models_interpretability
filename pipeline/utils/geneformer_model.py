"""
Geneformer V2 (104M) and scCello vocabulary, gene names and LoRA classifier.

Geneformer and scCello are plain BERTs, so the backbone is loaded with
transformers and the LoRA adapter with peft (see steps/_step7_geneformer.py).
The head on top is the same FlexMLP class as the scGPT one (utils/scgpt_model.py).
"""

import pickle
from pathlib import Path

import pandas as pd
from torch import nn

# The classification head reads the CLS token after the second-to-last encoder
# block, which is the cell embedding Geneformer's own EmbExtractor uses
EMB_HIDDEN_LAYER = -2

# scCello's cell embedding is the CLS token after the last encoder block
SCCELLO_EMB_HIDDEN_LAYER = -1


# Loading the Geneformer or scCello vocabulary (Ensembl ID -> token id)
def load_token_dictionary(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# Loading the token id -> gene symbol map of the genes in one dataset (gene_names.csv
# of scGPT, Geneformer or scCello)
def load_gene_names(path: Path) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["token_id"], df["gene"]))


# Backbone + FlexMLP head, the same classifier the LoRA was trained as
class GeneformerClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, emb_layer: int = EMB_HIDDEN_LAYER):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.emb_layer = emb_layer

    # Integrated Gradients feeds the token embeddings (inputs_embeds) instead of the tokens
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        out = self.backbone.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 inputs_embeds=inputs_embeds, output_hidden_states=True)
        # CLS token (position 0) of the embedding layer, one PD logit per cell
        return self.head(out.hidden_states[self.emb_layer][:, 0, :]).squeeze(1)
