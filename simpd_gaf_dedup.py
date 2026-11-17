"""
SimPD of the z-score gene sets (zero-shot+MLP vs LoRA) with a deduplicated GAF.

goa_human.gaf has one line per annotation *and* its evidence, so a gene repeats
the same GO term once per reference / evidence code (SNCA: 271 MF lines, 36
distinct terms, 198 of them "protein binding"). go3 keeps every line, so the
repeats weigh in both the gene's term list and the IC counts. Following JA's
proposal (meeting of 2026-11-16), each gene's GO terms are reduced per
sub-ontology with

    go3.reduce_redundant_terms(terms, threshold=1.0)   # method "lin", IC rank

which, at threshold 1.0, only merges terms of similarity 1 (in practice the
duplicates) and keeps one per group. The first GAF line of every kept
(gene, term) is written to a new GAF, which is then loaded as usual: the gene
annotations and the IC counter both come from it.

The table is rebuilt exactly as results/scgpt/tables/zscore_mlp_lora.csv
(macrotable rows Ranking pooled, Score IG, TopN zscore, scGPT, zero-shot+MLP /
LoRA, datasets A / B, 4 cell types): gene x reference-gene SimRel with
gene-level BMA, set-level BMA per sub-ontology (as gwas_similarity), the mean
over BP / CC / MF per reference, and
SimPD = mean(PD GWAS, PanelApp, ParkinsonsUK-UCL) - mean(AD, CKD, CAD).
n_genes and FGCell are copied from the original table (they do not depend on GO).

Needs go3 >= 2.0.0 (reduce_redundant_terms); 2.0.0 gives the same similarities
as 1.0.0 on the original GAF (--gaf original reproduces the original table).

    python simpd_gaf_dedup.py                  # dedup GAF -> zscore_mlp_lora_dedup.csv
    python simpd_gaf_dedup.py --gaf original   # check: the original table again
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import go3

from pipeline.postprocess import _go3_common as g3c
from pipeline.postprocess.gene_sets import CELL_TYPES, load_genes
from pipeline.utils.progress import progress

RESULTS = Path("results/scgpt")
TABLE = RESULTS / "tables" / "zscore_mlp_lora.csv"
OUT_DIR = RESULTS / "go3_dedup"
DISTANCE = "SimRel"
ONTOLOGIES = {"BP": "P", "CC": "C", "MF": "F"}
FORMS = {"mlp": "zero-shot+MLP", "lora": "LoRA"}
DATASETS = {"A": "dataset_a", "B": "dataset_b"}
CELLTYPE_LABELS = {"astrocytes": "astrocyte", "da_neurons": "neuron",
                   "microglia": "microglia", "oligodendrocytes": "oligodendrocyte"}
# Column label -> reference of gwas_similarity (as pipeline.macrotable.columns.similarity)
REFERENCES = {"PDGWAS": "parkinson", "PDGE": "panelapp", "PDGO": "parkinsonsuk_ucl",
              "AD": "alzheimer", "Kidney": "kidney", "CAD": "cad"}


# Reduce every gene's terms per sub-ontology and write the first GAF line of each kept (gene, term)
def write_dedup_gaf(threshold: float, method: str, out_gaf: Path) -> pd.DataFrame:
    go3.load_go_terms(str(g3c.OBO_PATH))
    annotations = go3.load_gaf(str(g3c.GAF_PATH))
    counter = go3.build_term_counter(annotations)

    # The terms go3 accepts for each (gene, aspect), in GAF order with their repeats
    # (go3 already leaves out NOT lines and obsolete terms)
    terms = {}
    for a in annotations:
        if not a.is_negated:
            terms.setdefault((a.db_object_symbol, a.aspect), []).append(a.go_id)

    kept, rows = set(), []
    for (gene, aspect), ids in progress(terms.items(), total=len(terms), desc="reduce", unit="gene-aspect"):
        reduced = go3.reduce_redundant_terms(ids, threshold=threshold, method=method, counter=counter)
        kept.update((gene, t) for t in reduced)
        rows.append({"gene": gene, "aspect": aspect, "n_lines": len(ids), "n_unique": len(set(ids)),
                     "n_kept": len(reduced)})
    stats = pd.DataFrame(rows)

    # Copy the header and the first line of every kept (gene, term), skipping NOT lines
    out_gaf.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(g3c.GAF_PATH) as src, open(out_gaf, "w") as dst:
        for line in src:
            if line.startswith("!"):
                dst.write(line)
                continue
            f = line.split("\t")
            key = (f[2], f[4])
            if "NOT" in f[3] or key not in kept or key in seen:
                continue
            seen.add(key)
            dst.write(line)
    print(f"GAF lines kept: {len(seen):,} of {len(annotations):,} accepted annotations -> {out_gaf}", flush=True)
    return stats


# Gene x reference-gene SimRel (gene-level BMA) for every gene of the 16 sets, one matrix per ontology
def pair_matrices(genes: list, refs: list, counter) -> dict:
    out = {}
    for ont in ONTOLOGIES:
        pairs = [(g, r) for g in genes for r in refs]
        sims = go3.compare_gene_pairs_batch(pairs, ont, DISTANCE, g3c.PAIR_GROUPWISE, counter)
        out[ont] = g3c.PairMatrix(genes, refs, np.asarray(sims).reshape(len(genes), len(refs)))
        print(f"  {ont}: {len(pairs):,} gene pairs", flush=True)
    return out


def build_table(gaf: Path) -> pd.DataFrame:
    go3.load_go_terms(str(g3c.OBO_PATH))
    counter = go3.build_term_counter(go3.load_gaf(str(gaf)))
    # The cached reference gene sets (identical in A and B)
    refs_dir = RESULTS / "dataset_a" / "gwas_similarity" / "refs"
    refs = {key: pd.read_csv(refs_dir / f"{key}_genes.csv")["gene"].astype(str).tolist()
            for key in REFERENCES.values()}

    sets = {}
    for ds, folder in DATASETS.items():
        for form in FORMS:
            for ct in CELL_TYPES:
                sets[(ds, form, ct)] = load_genes(RESULTS / folder, f"ig_{form}_scgpt", "z-score", ct)
    genes = sorted({g for s in sets.values() for g in s})
    all_refs = sorted({g for r in refs.values() for g in r})
    print(f"{len(sets)} gene sets, {len(genes)} genes, {len(all_refs)} reference genes", flush=True)
    pm = pair_matrices(genes, all_refs, counter)

    rows = []
    for (ds, form, ct), s in sets.items():
        row = {"Dataset": ds, "ModelForm": FORMS[form], "Celltype": CELLTYPE_LABELS[ct]}
        for label, key in REFERENCES.items():
            row[f"Sim{label}"] = np.mean([g3c.bma(pm[o].block(s, refs[key])) for o in ONTOLOGIES])
        row["SimPD"] = ((row["SimPDGWAS"] + row["SimPDGE"] + row["SimPDGO"]) / 3
                        - (row["SimAD"] + row["SimKidney"] + row["SimCAD"]) / 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gaf", choices=["dedup", "original"], default="dedup")
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--method", default="lin")
    p.add_argument("--out", type=Path, default=None, help="Output table (default: tables/zscore_mlp_lora_<gaf>.csv)")
    args = p.parse_args()
    go3.set_num_threads(int(os.environ.get("GO3_THREADS", 0)) or os.cpu_count() or 16)
    if not hasattr(go3, "reduce_redundant_terms"):
        raise SystemExit("go3 >= 2.0.0 is needed (reduce_redundant_terms)")

    if args.gaf == "dedup":
        gaf = OUT_DIR / f"goa_human_dedup_t{args.threshold:g}_{args.method}.gaf"
        stats = write_dedup_gaf(args.threshold, args.method, gaf)
        stats.to_csv(OUT_DIR / f"dedup_stats_t{args.threshold:g}_{args.method}.csv", index=False)
        print("Per (gene, aspect): lines", stats["n_lines"].sum(), "unique terms", stats["n_unique"].sum(),
              "kept", stats["n_kept"].sum(), "| distinct terms merged:",
              int((stats["n_unique"] - stats["n_kept"]).sum()), flush=True)
    else:
        gaf = g3c.GAF_PATH

    sims = build_table(gaf)
    # Same rows, order and columns as the original table, with the GO columns recomputed
    table = pd.read_csv(TABLE)
    keys = ["Dataset", "ModelForm", "Celltype"]
    table = table.drop(columns=[c for c in sims.columns if c not in keys]).merge(sims, on=keys, how="left")
    table = table[pd.read_csv(TABLE, nrows=0).columns]
    out = args.out or TABLE.with_name(f"zscore_mlp_lora_{args.gaf}.csv")
    table.to_csv(out, index=False)
    print("Saved:", out, flush=True)


if __name__ == "__main__":
    main()
