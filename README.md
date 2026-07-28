# SDT-CSE

SDT-CSE preserves SDT's multimodal encoder, hierarchical gated fusion,
unimodal emotion classifiers, hard-label supervision, and KL
self-distillation. It adds an optional normalized projection of the fused
representation, a cosine emotion classifier, and CircularCSE.

The existing `../SDT` implementation and data are not modified.

## Experiment modes

| Mode | Fusion classifier | Self-distillation | CircularCSE |
|---|---|---:|---:|
| `sdt` | Original linear classifier | Yes | No |
| `sdt_cosine` | Spherical projection + cosine classifier | Yes | No |
| `sdt_cse` | Spherical projection + cosine classifier | Yes | Yes |

All modes use the same text, audio, and visual classifiers. The fused
classifier remains the teacher for all three unimodal branches.

The comparisons have distinct purposes:

- `sdt_cosine - sdt` measures the effect of replacing the linear head.
- `sdt_cse - sdt_cosine` isolates CircularCSE.
- `sdt_cse - sdt` measures the complete proposed model change.

## Data split

The default feature file is:

```text
../SDT/data/iemocap_multimodal_features.pkl
```

Its `trainVid` and `testVid` fields are authoritative. Split membership is:

```python
validation_size = int(0.10 * len(trainVid))
validation = trainVid[:validation_size]
training = trainVid[validation_size:]
testing = testVid
```

Membership is not shuffled. Training batch order is shuffled after the split.
The best checkpoint is selected by validation weighted F1, with validation
fusion CE as the tie-breaker. The test loader is evaluated only once, after
the selected checkpoint has been restored. Validation data is not folded back
into training.

## Objective

For all modes, the preserved SDT objective is:

```text
fusion CE
+ text CE + audio CE + visual CE
+ KL(fusion || text) + KL(fusion || audio) + KL(fusion || visual)
```

The original six IEMOCAP class weights are applied to every CE term. The KL
terms use temperature-scaled fused teacher probabilities and unimodal student
log-probabilities. They are not multiplied by temperature squared, and the
teacher is not detached, matching the repository implementation.

For `sdt_cse`, the complete objective is:

```text
SDT objective + circular_weight * CircularCSE
```

The emotion order is:

```text
happy → excited → angry → frustrated → sad → neutral → happy
```

For the pickle's label IDs, the angle vector is:

```text
[0, 4π/3, 5π/3, 2π/3, π/3, π]
```

CircularCSE is computed from every ordered pair of valid projected fusion
embeddings in a minibatch. Padded utterances and diagonal self-pairs are
excluded.

## Commands

Run one primary SDT-CSE experiment:

```bash
cd SDT-CSE
python train.py \
  --experiment-mode sdt_cse \
  --circular-weight 0.1 \
  --seed 2024
```

Run the corresponding controls:

```bash
python train.py --experiment-mode sdt --seed 2024
python train.py --experiment-mode sdt_cosine --seed 2024
```

Run all three modes over ten initialization seeds and aggregate them:

```bash
bash exec_iemocap.sh
```

Run the predefined CircularCSE sensitivity analysis:

```bash
bash exec_iemocap_sweep.sh
```

Existing nonempty run directories are protected. Pass `--overwrite` only when
you intentionally want to replace files for the same condition and seed.

## Outputs

Each run is written below `SDT-CSE/results/<condition>/seed_<seed>/` and
contains:

- the complete configuration and exact split IDs;
- epoch-level training and validation metrics;
- the best validation checkpoint;
- final validation and single-pass test metrics;
- utterance-level test predictions;
- `features_train.npz`, `features_valid.npz`, and `features_test.npz`;
- fused and projected test representations;
- target, observed, and error similarity matrices;
- heatmaps and deterministic PCA plots.

The three `features_*.npz` files follow the established
`results/alv_IEMOCAP_20260630_054819/features_train.npz` contract:

```text
labels_emo
preds_emo
labels_sen
preds_sen
dialogue_ids
utterance_indices
utterance_ids
sentences
feature_l
feature_v
feature_a
feature_fusion
```

Cosine and CircularCSE modes also add `feature_embedding`, containing the
normalized projection used by their cosine classifier and by CircularCSE.
`feature_l`, `feature_v`, and `feature_a` are SDT's final text-, visual-, and
audio-oriented representations immediately before their unimodal
classifiers. `feature_fusion` is the final hierarchical gated fusion
representation.

SDT has no separate sentiment head. For schema compatibility,
`labels_sen` and `preds_sen` are derived deterministically from the emotion
IDs: sad/angry/frustrated are negative (`0`), neutral is neutral (`1`), and
happy/excited are positive (`2`).

All three files are exported after restoring the best validation checkpoint.
The training export uses a deterministic, non-shuffled loader; it does not
perform an optimizer step. Test inference still occurs exactly once.

Run `aggregate_results.py` to create per-seed, mean/std, and paired-difference
tables:

```bash
python aggregate_results.py --output-dir results
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover the circular target, loss gradients, padding masks,
self-distillation, the fixed split, fusion-head modes, and original SDT
feature/logit parity for the linear control.

## References

- Ma et al., *A Transformer-Based Model With Self-Distillation for Multimodal
  Emotion Recognition in Conversations*, IEEE TMM 2024.
- Yamauchi and Aizawa, *Mapping the Circumplex of Affect: Geometric Analysis
  of Emotion Representations via Hyperspherical Contrastive Learning*,
  ACL 2026.
