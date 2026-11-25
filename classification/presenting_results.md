# Presenting the classification results

How to read the outputs of `train_mlp.py` and `plot_results.py`, and how to present them in a talk or a paper.

## 1. What the numbers answer

The test set is four people (two PD, two control) the classifier never saw. Every number answers "does what the head learned from the training donors carry over to new donors?". The unit that generalises is the **donor**, not the cell: a test set of 4000 cells is still four people.

The zero-shot embedding + MLP is the baseline of the project. The paper asks how much fine-tuning (LoRA) adds to what the pretrained model already encodes, so this result is the reference that LoRA is measured against, not the goal itself.

## 2. The metrics

| Metric | What it says | Chance |
|---|---|---|
| Balanced accuracy | mean of sensitivity and specificity; the main metric, because the test sets do not have as many PD cells as control cells (A: 2942 vs 1487) | 0.5 |
| Sensitivity | fraction of PD cells called PD | |
| Specificity | fraction of control cells called control | |
| ROC AUC | how well the PD probability ranks PD cells above control cells, whatever the threshold | 0.5 |
| Donors correct | test donors whose cells have a mean PD probability on the right side of 0.5 | 2 of 4 by guessing |

Plain accuracy is misleading here: a head that calls every cell PD gets an accuracy of 0.66 in the test of Dataset A and a balanced accuracy of 0.5.

### The 95% CI and why it is not enough

`test_metrics.json` gives the balanced accuracy with a 95% CI computed over the cells (normal approximation; the one in the paper's Table S2). That interval only covers the randomness of which cells were sequenced. It treats the 4429 test cells of Dataset A as independent, but they come from four people, and cells of the same person look alike. The uncertainty that matters, which four people happen to be in test, is much larger and the cell-level CI does not show it. Hence:

- **Always show the per-donor result** (`test_donors.csv`, figure `test_donors`) next to the balanced accuracy.
- **Do not claim significance from the cell-level CI alone.** A CI that excludes 0.5 by a few thousandths says the cells were not classified at random, not that the disease signal generalises.
- **Repeat the training with several seeds** (`train_mlp.py --seed 42`, `43`, ...; five is enough) and report the mean over the seeds with its 95% CI (`plot_results.py` does it). This covers the randomness of training (initialisation, balancing, batch order), still not that of the donors.

### Validation is not a result

The validation set (also four donors) picks the epoch the head keeps. Its balanced accuracy is optimistic by construction and is never reported as the performance. The training curves (figure `training`) are for checking that training behaved: losses that go down, no head stuck on one class.

## 3. Worked example: the first run (2026-11-25, seed 42)

| Dataset | Model | Test balanced accuracy [95% CI] | Sensitivity | Specificity | Donors correct |
|---|---|---|---|---|---|
| A | scGPT | 0.542 [0.527, 0.557] | 0.407 | 0.677 | 2/4 |
| A | Geneformer | 0.565 [0.550, 0.581] | 0.631 | 0.500 | 2/4 |
| A | scCello | 0.445 [0.429, 0.460] | 0.512 | 0.377 | 2/4 |
| B | scGPT | 0.509 [0.500, 0.518] | 0.867 | 0.151 | 3/4 |
| B | Geneformer | 0.539 [0.532, 0.545] | 0.970 | 0.108 | 3/4 |
| B | scCello | 0.499 [0.492, 0.507] | 0.904 | 0.095 | 3/4 |

How to read it:

- **All six are close to chance.** The narrow CIs exclude 0.5 in four cases, but the donor table shows why that does not mean much: in Dataset A every model gets two of the four donors right, which is what guessing gives.
- **One donor can decide the whole result.** In Dataset B, control donor HBHF has 2739 of the 3096 control test cells, and the three models call it PD (mean PD probability 0.74 to 0.89). Specificity falls to 0.10 to 0.15 and the balanced accuracy to about 0.5, although three of the four donors are called correctly. Without the donor table, one would conclude that the models cannot recognise controls; with it, that they fail on one person.
- **Below chance is not "anti-learning".** scCello in Dataset A (0.445) calls control donor 2919 PD. With four test donors, a balanced accuracy below 0.5 most likely means that what the head learned from the training donors does not hold for these test donors.
- **Validation was much better than test.** In Dataset B, the heads kept for Geneformer and scCello had a validation balanced accuracy of 0.839 and 0.829 against 0.539 and 0.499 in test: the four validation donors happened to be easy. This is why the result is always the test.
- **The embeddings separate donors more than diagnosis** (figure `umap`): in Dataset A, Geneformer's UMAP has a separate cluster of control donor 2919's cells. A pretrained embedding encodes who the cell comes from (genotype, age, sample processing) at least as strongly as the disease, which is what a donor split exposes.

The conclusion of such a table is a sentence like: "From the zero-shot embedding alone, the MLP head did not generalise to unseen donors: its test balanced accuracy did not exceed 0.565 for any model in either dataset (see Fig. 2a, b), and no model classified more than three of the four test donors correctly (see Fig. 2c)." That is a valid and useful result: it is the baseline that LoRA has to improve on.

## 4. The figures

| Figure | What it shows | Where |
|---|---|---|
| `balanced_accuracy` | test balanced accuracy and 95% CI, one point per model, a panel per dataset, dashed line at chance | main figure |
| `test_donors` | PD probability of the test cells of each donor (box: quartiles; whiskers: 1.5 interquartile ranges), coloured by diagnosis | main figure, next to the previous one |
| `umap` | UMAP of each embedding coloured by diagnosis and by test donor | supplementary; qualitative only, a UMAP distorts distances and cluster sizes |
| `training` | train and validation loss, validation balanced accuracy, kept epoch (dotted) | supplementary |

Captions start with a bold title and describe every panel with its bold letter and what the error bars are, e.g.:

> **Fig. 2. Classification of PD astrocytes of unseen donors from zero-shot embeddings. a)** Test balanced accuracy of the MLP head on the embedding of each model in Dataset A; error bars, 95% CI (normal approximation over cells); dashed line, chance level. **b)** The same for Dataset B. **c)** PD probability of the test cells of each donor ...

## 5. Writing the result

The group writes papers in the style of `mlasdo_manuscript.docx`. The rules that matter most for results like these:

- **First person plural and active voice**: "We trained an MLP head on ...".
- **Every claim carries its number and where to find it**: "0.542 [0.527, 0.557] 95% CI (see Fig. 2a)". Numbers that are in no figure or table do not go in the text.
- **Summaries over several cases as a mean with its 95% CI**, not as "ranging from X to Y" (keep ranges for sizes and bounds).
- **One measure per idea.** If the balanced accuracy already supports a claim, do not add the accuracy, the F1 and the AUC for the same claim; they go to a supplementary table.
- **Exact verbs.** 0.565 versus 0.542 is "slightly exceeded", never "matched"; a value that went down "decreased", not "did not increase".
- **"Significant" always comes with its P-value and the test** (italic *P*). With four test donors, avoid significance claims between models: a cell-level test (e.g., McNemar's test on the paired predictions of two models) treats correlated cells as independent and overstates the evidence. Say what the data can and cannot show ("we are aware that four test donors per dataset ...").
- **Counts of cells and donors go in Methods**; Results only names the dataset and points to the Methods section.
- Abbreviations defined at first use; gene symbols in italics; "Fig." for figures.

A Results paragraph opens with what was asked and why, gives the values with their statistic and figure panel, interprets them against the literature and closes with a summary sentence:

> To establish what the pretrained representation already encodes about PD, we asked whether an MLP head trained on the zero-shot embedding of each model could classify the astrocytes of donors not seen in training. The test balanced accuracy was XXX [XXX, XXX] 95% CI for scGPT, XXX [XXX, XXX] for Geneformer and XXX [XXX, XXX] for scCello in Dataset A (see Fig. 2a), and ... in Dataset B (see Fig. 2b). At the donor level, the head classified XXX of the four test donors correctly with every model (see Fig. 2c) ... These values are in line with XXX *et al.*, who reported ... Together, these findings suggest that ...

## 6. Checklist before showing a result

1. Is it the test set (not validation), and are its donors absent from training? (`train_mlp.py` reads the split from the data; never mix the files.)
2. Is the balanced accuracy shown with its CI **and** the per-donor result?
3. With several seeds, is the mean over seeds reported, with the number of seeds?
4. Is every number in a figure or table, and does each sentence use one measure per idea?
5. Does every "significant" have its test and *P*, and is the comparison verb exact?
