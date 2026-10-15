"""
Frozen scGPT brain model + FlexMLP head.

Trimmed from the original pipeline_scgpt/utils/scgpt_model.py: keeps only
the model-loading and FlexMLP-head code used for inference (zero-shot and
LoRA-checkpoint loading).
"""

import json
from typing import List

import torch
import torchtext
from torch import nn

torchtext.disable_torchtext_deprecation_warning()

from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab

# Configuration for the run
from .. import config as C

_ACT_FNS = {"relu": nn.ReLU, "gelu": nn.GELU}

# Loading the scGPT vocab for the gene universe
def load_vocab() -> GeneVocab:
    vocab = GeneVocab.from_file(C.SCGPT_VOCAB)
    for s in C.SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    return vocab


# Loading the scGPT model configuration
def load_model_configs() -> dict:
    with open(C.SCGPT_ARGS) as f:
        return json.load(f)

# Function for loading the scGPT model though its library
def build_base_model(vocab: GeneVocab, model_configs: dict, n_cls: int = 1) -> TransformerModel:
    """Construct the base TransformerModel matching scGPT_brain pretraining."""
    return TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs.get("n_layers_cls", 3),
        n_cls=n_cls,
        vocab=vocab,
        dropout=model_configs.get("dropout", 0.2),
        pad_token=C.PAD_TOKEN,
        pad_value=C.PAD_VALUE,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        input_emb_style="continuous",
        n_input_bins=C.N_BINS,
        cell_emb_style="cls",
        ecs_threshold=0.8,
        explicit_zero_prob=False,
        use_fast_transformer=True,
        fast_transformer_backend="flash",
        pre_norm=model_configs.get("pre_norm", False),
    )

# The weights are scGPT_brain's by default, or those of the same model after Step 2.5
def load_pretrained_weights(model: TransformerModel, device: torch.device,
                            weights_path=C.SCGPT_WEIGHTS) -> None:
    # Load the .pt file
    pretrained = torch.load(weights_path, map_location=device)
    # Load of pytorch modules
    model_dict = model.state_dict()
    # Only load weights that match in name and shape (same dimensions) to the scGPT 
    matched = {k: v for k, v in pretrained.items()
               if k in model_dict and v.shape == model_dict[k].shape}
    # Update the model with the matched weights
    model_dict.update(matched)
    model.load_state_dict(model_dict)


# Creating custom MLP
class FlexMLP(nn.Module):
    def __init__(self, input_dim: int, n_layers: int, hidden_size: int,
                 dropout: float, activation: str, mlp_type: str = "constant",
                 n_classes: int = 1, norm: str = "none"):
        super().__init__()
        act_cls = _ACT_FNS[activation]
        layers: List[nn.Module] = []
        in_dim = input_dim
        current_size = hidden_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, current_size))
            if norm == "layernorm":
                layers.append(nn.LayerNorm(current_size))
            # For case the user wants to use no normalization
            elif norm != "none":
                raise ValueError(f"Unknown norm: {norm!r}")
            layers += [act_cls(), nn.Dropout(dropout)]
            in_dim = current_size
            # Dividing by two the side in case of "descendent", this is a prototype
            if mlp_type == "descendent":
                current_size = max(current_size // 2, 1)
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
