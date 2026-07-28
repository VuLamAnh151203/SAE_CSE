# SDT-CSE

SDT-CSE preserves SDT's multimodal encoder, hierarchical gated fusion,
hard-label supervision, and KL self-distillation. It adds optional cosine
emotion classifiers and CircularCSE.

The existing `../SDT` implementation and data are not modified.

## Experiment modes

| Mode | Fusion classifier | Unimodal classifiers | Self-distillation | CircularCSE |
|---|---|---|---:|---:|
| `sdt` | Original linear | Original linear | Yes | No |
| `sdt_cosine` | Spherical projection + cosine | Original linear | Yes | No |
| `sdt_cse` | Spherical projection + cosine | Original linear | Yes | Yes |
| `sdt_cse_all_cosine` | Spherical projection + cosine | Three spherical projections + cosine | Yes | Yes |
| `sdt_cse_fusion_only` | Spherical projection + cosine | None | No | Yes |

In `sdt_cse_all_cosine`, each final SDT text, audio, and visual
representation is processed by an independent head with the same structure
as the fusion head:

```text
Linear(H,H) -> GELU -> Dropout -> Linear(H,embedding_dim)
-> L2 normalize -> cosine classifier
```

The four heads have independent projection parameters, normalized class
weights, and learnable scales. The fused classifier remains the
self-distillation teacher for all three branches in every mode.

The comparisons have distinct purposes:

- `sdt_cosine - sdt` measures the effect of replacing the linear head.
- `sdt_cse - sdt_cosine` isolates CircularCSE.
- `sdt_cse - sdt` measures the complete proposed model change.
- `sdt_cse_all_cosine - sdt_cse` isolates replacing the three original
  unimodal classifiers with cosine classifiers.
- `sdt_cse_fusion_only - sdt_cse` measures the contribution of the three
  unimodal CE losses and self-distillation branches.

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

For all modes with unimodal classifiers, the preserved SDT objective is:

```text
fusion CE
+ text CE + audio CE + visual CE
+ KL(fusion || text) + KL(fusion || audio) + KL(fusion || visual)
```

The original six IEMOCAP class weights are applied to every CE term. The KL
terms use temperature-scaled fused teacher probabilities and unimodal student
log-probabilities. They are not multiplied by temperature squared, and the
teacher is not detached, matching the repository implementation.

For `sdt_cse` and `sdt_cse_all_cosine`, the complete objective is:

```text
SDT objective + circular_weight * CircularCSE
```

For `sdt_cse_fusion_only`, the encoders and hierarchical fusion are
unchanged, but no unimodal classifiers are constructed. Its objective is:

```text
fusion CE + circular_weight * CircularCSE
```

Consequently, all logged unimodal CE and KL components are exactly zero.

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

### Circular geometries

`--circular-geometry` selects the target angles independently of the model
architecture:

- `equal` is the original equally spaced six-emotion circle and remains the
  default.
- `nrc_vad` derives non-equally spaced angles from fixed NRC-VAD valence and
  arousal anchors around a configurable affect-space center.

The nonuniform version uses:

```text
theta_c = atan2(A_c - A_0, V_c - V_0) mod 2*pi
```

With the default center `(V_0, A_0) = (0.5, 0.5)`, the anchors and resulting
angles in label-ID order are:

| ID | Emotion | Valence | Arousal | Angle |
|---:|---|---:|---:|---:|
| 0 | happy | 0.960 | 0.732 | 26.764 degrees |
| 1 | sad | 0.052 | 0.288 | 205.324 degrees |
| 2 | neutral | 0.469 | 0.184 | 264.397 degrees |
| 3 | angry | 0.167 | 0.865 | 132.375 degrees |
| 4 | excited | 0.908 | 0.931 | 46.570 degrees |
| 5 | frustrated | 0.060 | 0.730 | 152.403 degrees |

Neutral has a defined angle because its NRC-VAD anchor is not exactly the
center. Configuration is rejected if a custom center exactly coincides with
any anchor, because `atan2(0, 0)` has no meaningful affective direction.
The selected angles and complete `6 x 6` target matrix are stored in every
checkpoint and in `circular_geometry.json`.

## Commands

Run one primary SDT-CSE experiment:

```bash
cd SDT-CSE
python train.py \
  --experiment-mode sdt_cse \
  --circular-weight 0.1 \
  --device cuda \
  --gpu-id 0 \
  --seed 2024
```

Run the non-equally spaced NRC-VAD version:

```bash
python train.py \
  --experiment-mode sdt_cse \
  --circular-geometry nrc_vad \
  --vad-center-valence 0.5 \
  --vad-center-arousal 0.5 \
  --circular-weight 0.1 \
  --device cuda \
  --gpu-id 0 \
  --seed 2024
```

This writes to:

```text
results/sdt_cse_nrc_vad_lambda_0.1/seed_2024/
```

Select another visible GPU by changing `--gpu-id`, for example:

```bash
python train.py --experiment-mode sdt_cse --device cuda --gpu-id 1
```

GPU IDs are zero-based indices among the devices visible to the process. If
`CUDA_VISIBLE_DEVICES` is set, `--gpu-id 0` means the first GPU in that
restricted list.

Run the corresponding controls:

```bash
python train.py --experiment-mode sdt --seed 2024
python train.py --experiment-mode sdt_cosine --seed 2024
python train.py --experiment-mode sdt_cse_all_cosine --seed 2024
python train.py --experiment-mode sdt_cse_fusion_only --seed 2024
```

Run all five modes over ten initialization seeds and aggregate them:

```bash
bash exec_iemocap.sh
```

For the launchers, select a GPU with the `GPU_ID` environment variable:

```bash
GPU_ID=1 bash exec_iemocap.sh
```

Run the predefined CircularCSE sensitivity analysis:

```bash
bash exec_iemocap_sweep.sh
```

Run all three CircularCSE architectures with NRC-VAD geometry over seeds
2024-2033:

```bash
GPU_ID=1 bash exec_iemocap_vad.sh
```

## PCA visualization

Visualize the pre-projection fused representation:

```bash
python visualize_multimodal_pca.py \
  --path results/sdt_cse_lambda_0.1/seed_2024/features_test.npz \
  --feature-key feature_fusion
```

Visualize the normalized embedding used by CircularCSE and the cosine
classifier:

```bash
python visualize_multimodal_pca.py \
  --path results/sdt_cse_lambda_0.1/seed_2024/features_test.npz \
  --feature-key feature_embedding
```

Use `--dimensions 3 --show` for an interactive 3-D plot. By default, the
command writes the PCA plot, emotion-centroid CSV, and reusable PCA-data NPZ
to:

```text
pca_dimension/<condition>/<seed>/
```

For example:

```text
pca_dimension/sdt_cse_lambda_0.1/seed_2024/
```

Pass `--output-dir` only when a different destination is needed.

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

All non-`sdt` modes also add `feature_embedding`, containing the normalized
fusion projection used by their fusion cosine classifier and, in the two
CircularCSE modes, by CircularCSE.
`sdt_cse_all_cosine` additionally stores `feature_l_embedding`,
`feature_a_embedding`, and `feature_v_embedding`, which are the normalized
text, audio, and visual projections used by their respective cosine
classifiers.
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
