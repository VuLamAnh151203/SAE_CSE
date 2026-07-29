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
| `sdt_cse_all_modal_cse` | Spherical projection + cosine | Three spherical projections + cosine | Yes | Fusion + all three modalities |
| `sdt_cse_fusion_only` | Spherical projection + cosine | None | No | Yes |
| `sdt_cse_learnable_angles` | Spherical projection + cosine | Original linear | Yes | Learnable ordered angles |

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
`sdt_cse_all_modal_cse` uses the identical four-head architecture and adds
CircularCSE independently to the fusion, text, audio, and visual projected
embeddings.

The comparisons have distinct purposes:

- `sdt_cosine - sdt` measures the effect of replacing the linear head.
- `sdt_cse - sdt_cosine` isolates CircularCSE.
- `sdt_cse - sdt` measures the complete proposed model change.
- `sdt_cse_all_cosine - sdt_cse` isolates replacing the three original
  unimodal classifiers with cosine classifiers.
- `sdt_cse_all_modal_cse - sdt_cse_all_cosine` isolates applying
  CircularCSE to the three unimodal projected embeddings.
- `sdt_cse_fusion_only - sdt_cse` measures the contribution of the three
  unimodal CE losses and self-distillation branches.
- `sdt_cse_learnable_angles - sdt_cse` isolates learning ordered class
  angles while preserving the complete standard SDT-CSE architecture.

## Internal spherical residuals

Every experiment mode supports an orthogonal residual option:

```text
--sdt-residual-update {standard,spherical}
```

`standard` is the default and preserves the original SDT additions. The
`spherical` option replaces the attention and MLP additions inside each of
the nine intra/inter-modal transformer branches with:

```text
normalize(
  (1 - alpha) * normalize(current)
  + alpha * normalize(proposal)
)
```

Each branch owns one scalar attention alpha and one scalar MLP alpha: 18
learned gates in total. Valid utterance states remain unit norm after both
updates, while padded states remain zero. Positional/speaker embeddings,
attention, LayerNorm, MLP, fusion gates, classifiers, and losses otherwise
retain their existing definitions.

Configure the two initial gates with:

```text
--spherical-attention-alpha-init 0.1
--spherical-mlp-alpha-init 0.1
```

Gate logits use the normal learning rate and zero weight decay. Spherical
residuals add no new loss term and can be combined with every experiment
mode and every loss ablation.

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

An explicit original-SDT-style protocol is also available:

```bash
--selection-protocol test
```

It trains on every `trainVid` dialogue, creates no validation split, evaluates
`testVid` after every epoch, and selects the earliest epoch attaining the
highest test weighted F1 after rounding to two decimals, matching the original
SDT selection code. This protocol uses test information for model selection
and therefore produces test-selected, optimistically biased estimates. It is
provided only for direct behavioral comparison with the original SDT
launcher. Its result directory ends in `_test_selected`.
Unlike the original script, this implementation saves and restores the
test-selected checkpoint so its predictions and embeddings correspond to the
reported selected epoch.

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

For `sdt_cse_all_modal_cse`, the objective is:

```text
SDT objective
+ circular_weight * (
    CircularCSE(fusion_embedding)
    + CircularCSE(text_embedding)
    + CircularCSE(audio_embedding)
    + CircularCSE(visual_embedding)
  )
```

The four CircularCSE terms share the selected fixed circular geometry and
same-class margin but are calculated independently.
They are logged as `fusion_circular_cse`, `text_circular_cse`,
`audio_circular_cse`, and `visual_circular_cse`.
`unimodal_circular_cse` is the sum of the three modality terms and
`total_circular_cse` is the sum of all four. The legacy `circular_cse`
field remains the fusion term for compatibility.

For `sdt_cse_fusion_only`, the encoders and hierarchical fusion are
unchanged, but no unimodal classifiers are constructed. Its objective is:

```text
fusion CE + circular_weight * CircularCSE
```

Consequently, all logged unimodal CE and KL components are exactly zero.

For `sdt_cse_learnable_angles`, the complete objective is:

```text
SDT objective
+ circular_weight * CircularCSE(learned angles)
+ angle_weight * sum((learned angle - prior angle)^2)
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

### Circular geometries

`--circular-geometry` selects the target angles independently of the model
architecture:

- `equal` is the original equally spaced six-emotion circle.
- `nrc_vad` derives non-equally spaced angles from fixed NRC-VAD valence and
  arousal anchors around a configurable affect-space center.

Fixed-angle modes default to `equal`. The learnable-angle mode defaults to
`nrc_vad`; either prior can be selected explicitly.

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

### Prior-regularized learnable angles

`sdt_cse_learnable_angles` fixes happy at zero and learns six positive gaps
in this immutable order:

```text
happy -> excited -> angry -> frustrated -> sad -> neutral -> happy
```

The gaps are parameterized and normalized as:

```text
g = softplus(raw_gaps)
normalized_gaps = 2*pi*g/sum(g)
```

This guarantees positive gaps, a total circumference of `2*pi`, and stable
class ordering. The selected prior initializes the gaps and penalizes angle
displacement. `raw_gaps` uses the normal learning rate but zero optimizer
weight decay.

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

Run the prior-regularized learnable-angle version:

```bash
python train.py \
  --experiment-mode sdt_cse_learnable_angles \
  --circular-geometry nrc_vad \
  --circular-weight 0.1 \
  --angle-weight 0.1 \
  --device cuda \
  --gpu-id 0 \
  --seed 2024
```

This writes to:

```text
results/sdt_cse_learnable_angles_nrc_vad_lambda_0.1_angle_0.1/seed_2024/
```

Run learnable equal-initialized angles with internal spherical residuals:

```bash
python train.py \
  --experiment-mode sdt_cse_learnable_angles \
  --selection-protocol validation \
  --circular-geometry equal \
  --circular-weight 0.1 \
  --angle-weight 0.1 \
  --sdt-residual-update spherical \
  --spherical-attention-alpha-init 0.1 \
  --spherical-mlp-alpha-init 0.1 \
  --device cuda \
  --gpu-id 0 \
  --seed 2024
```

This writes to:

```text
results/sdt_cse_learnable_angles_equal_lambda_0.1_angle_0.1_spherical_residual_a0.1_m0.1/seed_2024/
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
python train.py --experiment-mode sdt_cse_all_modal_cse --seed 2024
python train.py --experiment-mode sdt_cse_fusion_only --seed 2024
python train.py --experiment-mode sdt_cse_learnable_angles --seed 2024
```

Run standard SDT-CSE with the original SDT test-selection behavior:

```bash
python train.py \
  --experiment-mode sdt_cse \
  --selection-protocol test \
  --epochs 150 \
  --device cuda \
  --gpu-id 0 \
  --seed 2024
```

This writes to:

```text
results/sdt_cse_lambda_0.1_test_selected/seed_2024/
```

Run that protocol over seeds 2024-2033:

```bash
GPU_ID=0 bash exec_iemocap_test_selected.sh
```

Run the primary validation-selected spherical-residual experiment over seeds
2024-2033:

```bash
GPU_ID=0 bash exec_iemocap_spherical_residual.sh
```

Run the separately named original-SDT-style test-selected comparison:

```bash
GPU_ID=0 bash exec_iemocap_spherical_residual_test_selected.sh
```

Run the validation-selected alpha sensitivity analysis for
`alpha in {0.05, 0.1, 0.2}`:

```bash
GPU_ID=0 bash exec_iemocap_spherical_alpha_sweep.sh
```

The predefined main initialization is `0.1`; the sweep never selects alpha
using test performance.

Run all six modes over ten initialization seeds and aggregate them:

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

Run the learnable-angle prior-weight sensitivity analysis:

```bash
GPU_ID=1 bash exec_iemocap_angle_sweep.sh
```

This runs `angle_weight` values `0.01`, `0.1`, and `1.0` over seeds
2024-2033. Set `ANGLE_PRIOR=equal` to use the equal-spacing prior instead of
the default NRC-VAD prior.

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
- epoch-level training and selection-split metrics;
- the selected checkpoint and its selection protocol;
- final metrics and representations from the restored checkpoint;
- utterance-level test predictions;
- `features_train.npz`, `features_valid.npz`, and `features_test.npz`;
- fused and projected test representations;
- target, observed, and error similarity matrices;
- heatmaps and deterministic PCA plots.

For `--selection-protocol test`, `features_train.npz` covers all of
`trainVid`, while `features_valid.npz` is a schema-compatible empty archive
because no validation set exists. `features_test.npz` contains the restored
test-selected checkpoint's test representations.

Learnable-angle runs additionally contain:

- `angle_history.csv` with epoch-level angles and consecutive gaps;
- `learned_circular_geometry.json` from the selected checkpoint;
- prior angles, learned angles, gaps, offsets, and target similarities in
  the checkpoint and summary.

Spherical-residual runs additionally contain:

- `residual_gate_history.csv` with all 18 effective gates per epoch;
- `spherical_residual_diagnostics.json` with per-split norm and angular
  movement statistics for every attention and MLP update;
- residual type, gate initialization, and selected gate values in the
  checkpoint and summary.

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
fusion projection used by their fusion cosine classifier and, in CircularCSE
modes, by CircularCSE.
`sdt_cse_all_cosine` and `sdt_cse_all_modal_cse` additionally store
`feature_l_embedding`, `feature_a_embedding`, and `feature_v_embedding`,
which are the normalized text, audio, and visual projections used by their
respective cosine classifiers. In `sdt_cse_all_modal_cse`, these three
projected embeddings also receive their own CircularCSE terms.
`feature_l`, `feature_v`, and `feature_a` are SDT's final text-, visual-, and
audio-oriented representations immediately before their unimodal
classifiers. `feature_fusion` is the final hierarchical gated fusion
representation.

SDT has no separate sentiment head. For schema compatibility,
`labels_sen` and `preds_sen` are derived deterministically from the emotion
IDs: sad/angry/frustrated are negative (`0`), neutral is neutral (`1`), and
happy/excited are positive (`2`).

All three files are exported after restoring the selected checkpoint. The
training export uses a deterministic, non-shuffled loader and performs no
optimizer step. Under validation selection, test inference occurs exactly
once. Under original-SDT-style test selection, test is evaluated every epoch
and once more from the restored selected checkpoint for aligned artifacts.

Run `aggregate_results.py` to create per-seed, mean/std, and paired-difference
tables:

```bash
python aggregate_results.py --output-dir results
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover fixed and learnable circular targets, spherical residual
norms, padding, gate gradients and restoration, ordered positive angle gaps,
self-distillation, split lifecycles, fusion-head modes, and original SDT
feature/logit parity for the standard-residual linear control.

## References

- Ma et al., *A Transformer-Based Model With Self-Distillation for Multimodal
  Emotion Recognition in Conversations*, IEEE TMM 2024.
- Yamauchi and Aizawa, *Mapping the Circumplex of Affect: Geometric Analysis
  of Emotion Representations via Hyperspherical Contrastive Learning*,
  ACL 2026.
