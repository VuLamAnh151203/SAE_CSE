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
| `sdt_cse_learnable_angles_confusion_gap` | Spherical projection + cosine | Original linear | Yes | Learnable ordered angles with minimum confusion gaps |
| `sdt_cse_confusion_margin` | Spherical projection + cosine | Original linear | Yes | Fixed 75-degree confusion gaps with weighted confused pairs |
| `sdt_cse_bilevel_confusion_gap` | Spherical projection + cosine | Original linear | Yes | Shared confusion gap learned from validation classification |
| `sdt_cse_bilevel_all_gaps` | Spherical projection + cosine | Original linear | Yes | Six hard-floor ordered gaps learned from validation or training |

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
- `sdt_cse_learnable_angles_confusion_gap - sdt_cse_learnable_angles`
  measures whether explicitly widening happy–excited and
  angry–frustrated improves their classification.

- `sdt_cse_confusion_margin - sdt_cse` measures the combined effect of a
  fixed constrained geometry, stronger CircularCSE supervision for the two
  confused pairs, and a direct confusion-aware cosine-classification margin.

- `sdt_cse_bilevel_confusion_gap - sdt_cse_confusion_margin` replaces the
  fixed gap by one bounded shared gap whose hypergradient comes from
  validation fusion classification.
- `sdt_cse_bilevel_all_gaps - sdt_cse_bilevel_confusion_gap` removes the
  equal-confusion/equal-remaining-gap restriction and validation-learns all
  six consecutive gaps while preserving their circular order.

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

For `sdt_cse_learnable_angles_confusion_gap`, equal spacing is mandatory at
initialization and the objective additionally contains:

```text
confusion_gap_weight * (
  relu(75 degrees - gap(happy, excited))^2
  + relu(75 degrees - gap(angry, frustrated))^2
)
```

The ordered positive-gap parameterization remains unchanged, so the circular
order and total circumference are preserved. Increasing these two gaps
redistributes the remaining circumference across the other four gaps.

For `sdt_cse_confusion_margin`, the class angles are fixed rather than
learned. In circular order, the six consecutive gaps are:

```text
[75, 52.5, 75, 52.5, 52.5, 52.5] degrees
```

The first and third gaps are happy-to-excited and angry-to-frustrated.
`--confused-cse-pair-weight` applies symmetrically to every ordered
cross-class CircularCSE pair belonging to either confused pair. The weighted
mean is normalized by the sum of pair weights.

The mode also adds a direct raw-cosine classification constraint:

```text
confusion_classification_weight * mean(
  relu(
    confusion_classification_margin
    - (true_class_cosine - confused_class_cosine)
  )
)
```

Only gold happy, excited, angry, and frustrated utterances enter this term.
It directly updates the projected fusion embedding and cosine classifier
weights. Defaults are pair weight `5`, cosine margin `0.1`, and
classification-margin weight `0.1`. Original unimodal CE and
self-distillation remain active.

For `sdt_cse_bilevel_confusion_gap`, one shared scalar controls both
happy-excited and angry-frustrated gaps:

```text
gap = lower + (upper - lower) * sigmoid(raw_gap)
other_gap = (360 - 2 * gap) / 4
```

Defaults are a range of 70-110 degrees and initialization at 90 degrees.
The ordinary Adam optimizer updates every SDT/model parameter except
`raw_gap`. A separate zero-weight-decay Adam optimizer updates only
`raw_gap`.

The gap update uses a one-step DARTS finite-difference hypergradient:

```text
training objective at temporarily perturbed model weights
-> Hessian-vector product
-> validation fusion CE + validation confusion margin
-> bounded shared gap
```

The outer validation objective contains no CircularCSE, preventing the
target angle from merely adapting to the current validation geometry. Model
weights are never directly stepped using validation gradients. The mode
requires `--selection-protocol validation`, never accesses test during
training, and restores the best validation-weighted-F1 checkpoint before
the single test evaluation.

This method is approximate bilevel optimization because the hypergradient
differentiates through one virtual SGD step while the actual model optimizer
is Adam. It performs three additional forward/backward evaluations per
training minibatch and is therefore substantially slower than fixed-angle
training.

For `sdt_cse_bilevel_all_gaps`, every consecutive gap has its own learned
allocation:

```text
gap_c = minimum_gap
        + (360 - 6 * minimum_gap) * softmax(raw_gaps)_c
```

The six gaps are ordered as happy-to-excited, excited-to-angry,
angry-to-frustrated, frustrated-to-sad, sad-to-neutral, and
neutral-to-happy. This guarantees that every gap is strictly larger than the
configured minimum, their sum is exactly 360 degrees, and the class order
cannot change. Equal initialization starts all gaps at 60 degrees.

The all-gap outer objective adds:

```text
bilevel_gap_prior_weight
* sum((learned_gap - initialization_gap)^2)
```

The default minimum is 20 degrees and the prior weight is `0.01`. Set the
prior weight to zero for fully validation-driven spacing while retaining the
hard minimum. The validation classification objective remains fusion CE plus
the confused-pair cosine margin; validation weighted F1 remains the
checkpoint-selection metric. The squared prior penalty is computed in
radians internally.

Select where the six gap parameters receive gradients with:

```text
--all-gap-learning-source {validation,training}
```

`validation` is the default bilevel method. `training` keeps the identical
hard-floor parameterization, initialization, weighted confused-pair
CircularCSE, direct confusion-classification margin, and gap prior, but puts
the raw gaps in the ordinary training Adam optimizer with zero weight decay.
In training-source mode no validation minibatch or hypergradient is used;
validation only selects the checkpoint. The gaps use the normal `--lr`;
`--bilevel-angle-learning-rate`, `--bilevel-inner-step-size`,
`--bilevel-hvp-radius`, `--bilevel-outer-confusion-weight`, and
`--bilevel-angle-gradient-clip` are inactive.

NRC-VAD initialization is also available with
`--bilevel-all-gaps-initialization nrc_vad`, but its smallest prior gap is
about 19.8 degrees. Therefore, set
`--bilevel-minimum-class-gap-degrees` below that value, such as 10 degrees.
The program rejects an initialization that violates the configured floor.

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
- `confusion_separated` fixes the two predefined confusion gaps to
  `--minimum-confusion-gap-degrees` and divides the remaining circumference
  equally over the other four positive gaps.

Fixed-angle modes default to `equal`. The learnable-angle mode defaults to
`nrc_vad`; either prior can be selected explicitly. The
`sdt_cse_confusion_margin` and the shared-gap bilevel mode require
`confusion_separated`; the shared-gap bilevel mode replaces those fixed
angles dynamically. The all-gap bilevel mode defaults to `equal`
initialization and then learns all six gaps dynamically.

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
python train.py \
  --experiment-mode sdt_cse_learnable_angles_confusion_gap \
  --circular-geometry equal \
  --minimum-confusion-gap-degrees 75 \
  --confusion-gap-weight 0.1 \
  --seed 2024
python train.py \
  --experiment-mode sdt_cse_confusion_margin \
  --selection-protocol validation \
  --circular-geometry confusion_separated \
  --minimum-confusion-gap-degrees 75 \
  --circular-weight 0.1 \
  --confused-cse-pair-weight 5 \
  --confusion-classification-margin 0.1 \
  --confusion-classification-weight 0.1 \
  --device cuda \
  --gpu-id 0 \
  --seed 2024
python train.py \
  --experiment-mode sdt_cse_bilevel_confusion_gap \
  --selection-protocol validation \
  --circular-weight 0.1 \
  --confused-cse-pair-weight 5 \
  --confusion-classification-margin 0.1 \
  --confusion-classification-weight 0.1 \
  --bilevel-gap-minimum-degrees 70 \
  --bilevel-gap-maximum-degrees 110 \
  --bilevel-gap-initial-degrees 90 \
  --bilevel-angle-learning-rate 0.001 \
  --bilevel-inner-step-size 0.0001 \
  --bilevel-hvp-radius 0.01 \
  --bilevel-outer-confusion-weight 0.1 \
  --device cuda \
  --gpu-id 0 \
  --seed 2024
python train.py \
  --experiment-mode sdt_cse_bilevel_all_gaps \
  --selection-protocol validation \
  --all-gap-learning-source validation \
  --circular-weight 0.1 \
  --confused-cse-pair-weight 5 \
  --confusion-classification-margin 0.1 \
  --confusion-classification-weight 0.1 \
  --bilevel-all-gaps-initialization equal \
  --bilevel-minimum-class-gap-degrees 20 \
  --bilevel-gap-prior-weight 0.01 \
  --bilevel-angle-learning-rate 0.001 \
  --bilevel-inner-step-size 0.0001 \
  --bilevel-hvp-radius 0.01 \
  --bilevel-outer-confusion-weight 0.1 \
  --device cuda \
  --gpu-id 0 \
  --seed 2024
```

Run the same six-gap model while learning gaps only from training:

```bash
python train.py \
  --experiment-mode sdt_cse_bilevel_all_gaps \
  --selection-protocol validation \
  --all-gap-learning-source training \
  --circular-weight 0.1 \
  --confused-cse-pair-weight 5 \
  --confusion-classification-margin 0.1 \
  --confusion-classification-weight 0.1 \
  --bilevel-all-gaps-initialization equal \
  --bilevel-minimum-class-gap-degrees 20 \
  --bilevel-gap-prior-weight 0.01 \
  --device cuda \
  --gpu-id 0 \
  --seed 2024
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

Run the validation-selected minimum-gap weight sweep with
`confusion_gap_weight in {0.01, 0.1, 1.0}`, equal initialization, and a
minimum gap of 75 degrees:

```bash
GPU_ID=0 bash exec_iemocap_confusion_gap_sweep.sh
```

Choose the predefined gap weight from mean validation weighted F1, using the
validation happy–excited and angry–frustrated pair F1 values as diagnostics.
Do not choose it from test performance. Aggregation now writes validation
means and standard deviations alongside test results.

The sweep uses standard residual updates by default. To combine it with
internal spherical residuals:

```bash
GPU_ID=0 RESIDUAL_UPDATE=spherical \
  bash exec_iemocap_confusion_gap_sweep.sh
```

Run the recommended validation-selected confusion-margin sweep with
`confused_cse_pair_weight in {2, 5, 10}`:

```bash
GPU_ID=0 bash exec_iemocap_confusion_margin_sweep.sh
```

It uses the fixed 75-degree geometry, classification margin `0.1`,
classification-margin weight `0.1`, and standard SDT residual updates.

Run the validation-learned shared-gap experiment over seeds 2024-2033:

```bash
GPU_ID=0 bash exec_iemocap_bilevel_confusion_gap.sh
```

This starts both confusion gaps at 90 degrees and constrains them to
70-110 degrees. Because each training minibatch requires a validation
hypergradient calculation, expect this launcher to take several times
longer than the fixed-angle experiment.

Run the validation-learned six-gap experiment over seeds 2024-2033:

```bash
GPU_ID=0 bash exec_iemocap_bilevel_all_gaps.sh
```

It starts from six equal 60-degree gaps, enforces a 20-degree minimum for
each gap, and records all six learned values in `angle_history.csv`,
`bilevel_gap_history.csv`, checkpoints, per-seed summaries, and aggregate
CSV files.

The launcher defaults to validation gap learning. Run the training-only
counterpart with:

```bash
GPU_ID=0 GAP_LEARNING_SOURCE=training \
  bash exec_iemocap_bilevel_all_gaps.sh
```

Run all experiment modes over ten initialization seeds and aggregate them:

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

Confusion-gap runs also record the selected happy–excited and
angry–frustrated gaps, minimum gap, gap penalty, and gap weight in the
checkpoint, angle history, learned geometry, and summary. Every experiment
reports separate pair macro/weighted F1 and direct mutual-confusion rates for
both predefined pairs.

Confusion-margin runs record the fixed angles, exact pair gaps, target
similarity matrix, confused-pair CSE weight, cosine-classification margin,
classification-margin weight, and independently logged
`confusion_classification_margin` loss in checkpoints and summaries.

Bilevel confusion-gap runs additionally record:

- `bilevel_gap_history.csv` with epoch-level gap and hypergradient metrics;
- lower, upper, initial, and selected gap values;
- the outer angle optimizer state in every selected checkpoint;
- outer validation classification loss, scalar hypergradient, HVP radius,
  virtual inner-step size, and angle learning rate;
- the selected checkpoint's dynamic target matrix in geometry reports.

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
