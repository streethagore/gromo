# Hidden-representation visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans skill to implement this plan task-by-task.

**Goal:** Add a reproducible `experimental_grow` reference experiment that records a configured GroMo hidden layer on a fixed probe set and exports a temporally aligned 2D PCA animation.

**Architecture:** Keep all visualization code in `experimental_grow`: a recorder captures named-module outputs at explicit training boundaries, a global PCA projector gives every frame one coordinate system, and a renderer produces artifacts. `hydra_script/train_and_grow.py` merely orchestrates those components through an opt-in Hydra config; no GroMo module or greedy-learning code changes.

**Tech Stack:** Python 3.10+, PyTorch, Hydra/OmegaConf, Matplotlib with Pillow GIF writer, MLflow, pytest.

---

## Preconditions and invariants

- Work in a fresh feature branch or dedicated worktree of `/Users/strivaud/Projects/research/experimental_grow`; it currently contains unrelated uncommitted files, which must not be staged or modified.
- Version 1 accepts only a rank-2 `(probe_samples, features)` layer output. The selected layer must retain its feature width across captured frames. Use a non-growing MLP layer while another layer grows for the first reference run.
- The test loader is the source of a fixed, label-stratified probe batch. It is never a shuffled training batch.
- PCA is fit once on the concatenated raw snapshots. Do not run PCA separately per frame.
- The first artifact format is GIF plus raw `.pt` snapshots and a JSON manifest. MP4, t-SNE, MDS, autoencoders, CNN reductions, and width-changing layers remain follow-up work.

### Task 1: Add the failing configuration and recorder-contract tests

**Files:**
- Create: `experimental_grow/tests/test_representation_visualization.py`
- Create: `experimental_grow/hydra_script/tests/test_visualization_config.py`

**Step 1: Write the failing unit tests for layer resolution and snapshot capture**

```python
import pytest
import torch

from tools.representation_visualization import RepresentationRecorder


def test_capture_records_cpu_rank_two_features_and_restores_training_mode() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    model.train()
    recorder = RepresentationRecorder(model=model, layer_name="0")

    frame = recorder.capture(
        inputs=torch.ones(5, 3),
        labels=torch.tensor([0, 1, 0, 1, 0]),
        step=3,
        phase="training",
        metrics={"loss": 0.5},
    )

    assert frame.features.shape == (5, 4)
    assert frame.features.device.type == "cpu"
    assert model.training is True
    assert frame.step == 3


def test_capture_rejects_missing_layer_name() -> None:
    with pytest.raises(ValueError, match="does not resolve"):
        RepresentationRecorder(torch.nn.Linear(3, 2), layer_name="missing")
```

**Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_representation_visualization.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.representation_visualization'`.

**Step 3: Write the failing Hydra-composition test**

```python
from hydra import compose, initialize_config_dir


def test_visualization_default_is_disabled() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir="/Users/strivaud/Projects/research/experimental_grow/hydra_script/configs",
    ):
        cfg = compose(config_name="config")

    assert cfg.visualization.enabled is False
    assert cfg.visualization.layer_name == ""
```

**Step 4: Run the configuration test to verify it fails**

Run: `uv run pytest hydra_script/tests/test_visualization_config.py -q`

Expected: FAIL because the `visualization` config group does not yet exist.

**Step 5: Commit the tests**

```bash
git add tests/test_representation_visualization.py hydra_script/tests/test_visualization_config.py
git commit -m "test: specify representation visualization contracts"
```

### Task 2: Implement fixed probe selection and one-shot representation capture

**Files:**
- Create: `experimental_grow/tools/representation_visualization.py`
- Modify: `experimental_grow/tests/test_representation_visualization.py`

**Step 1: Add failing tests for deterministic stratified selection and invalid hook output**

```python
def test_select_probe_batch_is_stratified_and_seeded() -> None:
    loader = make_loader(labels=[0, 0, 0, 1, 1, 1], batch_size=2)

    first = select_stratified_probe_batch(loader, maximum_samples=4, seed=7)
    second = select_stratified_probe_batch(loader, maximum_samples=4, seed=7)

    assert torch.equal(first.inputs, second.inputs)
    assert first.labels.bincount().tolist() == [2, 2]


def test_capture_rejects_non_tensor_hook_output() -> None:
    class TupleLayer(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return inputs, inputs

    with pytest.raises(ValueError, match="must be a tensor"):
        RepresentationRecorder(torch.nn.Sequential(TupleLayer()), "0").capture(
            torch.ones(2, 3), torch.tensor([0, 1]), 0, "initial", {}
        )
```

**Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/test_representation_visualization.py -q`

Expected: FAIL because probe selection and hook-output validation are not implemented.

**Step 3: Implement the minimal public data types and APIs**

In `tools/representation_visualization.py`, add documented dataclasses:

```python
@dataclass(frozen=True)
class ProbeBatch:
    """Fixed ordered inputs and labels used in every visualization frame."""
    inputs: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class RepresentationFrame:
    """One hidden-representation snapshot and immutable render metadata."""
    features: torch.Tensor
    labels: torch.Tensor
    step: int
    phase: str
    metrics: dict[str, float]
```

Implement `select_stratified_probe_batch(dataloader, maximum_samples, seed)` by
collecting CPU examples in deterministic dataloader order, grouping scalar class
labels, and sampling an equal per-class quota using a local
`torch.Generator().manual_seed(seed)`. Fill unused quota round-robin from the
remaining classes. Raise `ValueError` for an empty loader, non-rank-one labels,
or `maximum_samples < 2`.

Implement `RepresentationRecorder` with these rules:

- resolve `layer_name` against `dict(model.named_modules())` in `__init__`;
- register its forward hook inside `capture`, remove it in `finally`, and retain
  only the first hook value;
- switch the model to `eval()` only for the probe forward pass, use
  `torch.inference_mode()`, then restore the original `.training` value;
- reject non-tensor output, output other than rank 2, and a sample-count mismatch;
- save `output.detach().to(device="cpu", dtype=torch.float32).clone()` and clone
  labels to CPU; never retain a graph or GPU activation.

**Step 4: Run the focused tests to verify they pass**

Run: `uv run pytest tests/test_representation_visualization.py -q`

Expected: PASS.

**Step 5: Run linting for the new module**

Run: `uv run ruff check tools/representation_visualization.py tests/test_representation_visualization.py`

Expected: PASS.

**Step 6: Commit the capture component**

```bash
git add tools/representation_visualization.py tests/test_representation_visualization.py
git commit -m "feat: capture fixed hidden representation probes"
```

### Task 3: Implement globally aligned PCA projection

**Files:**
- Modify: `experimental_grow/tools/representation_visualization.py`
- Modify: `experimental_grow/tests/test_representation_visualization.py`

**Step 1: Add failing alignment and width-validation tests**

```python
def test_global_pca_preserves_one_coordinate_system_for_all_frames() -> None:
    frames = [
        make_frame([[0.0, 0.0, 9.0], [1.0, 0.0, 9.0]], step=0),
        make_frame([[0.0, 1.0, 9.0], [1.0, 1.0, 9.0]], step=1),
    ]

    projected = GlobalPCAProjector().fit_transform(frames)

    assert projected[1].coordinates[0, 1] != projected[0].coordinates[0, 1]
    assert GlobalPCAProjector().fit_transform(frames) == projected


def test_global_pca_rejects_feature_width_change() -> None:
    with pytest.raises(ValueError, match="feature width"):
        GlobalPCAProjector().fit_transform([
            make_frame([[0.0, 1.0]], step=0),
            make_frame([[0.0, 1.0, 2.0]], step=1),
        ])
```

Compare tensors with `torch.testing.assert_close`, not `==`, in the final test.

**Step 2: Run the projector tests to verify they fail**

Run: `uv run pytest tests/test_representation_visualization.py -k pca -q`

Expected: FAIL because `GlobalPCAProjector` does not exist.

**Step 3: Implement `GlobalPCAProjector` and projected frame types**

Use CPU `torch.linalg.svd` for the small reference probe matrix; this avoids a
scikit-learn dependency and produces a deterministic full decomposition for the
test sizes. Concatenate every frame's features, subtract one global mean, take
the first two rows of `Vh`, and project every frame with the same components.

```python
centered = all_features - mean
_, _, right_singular_vectors = torch.linalg.svd(centered, full_matrices=False)
components = right_singular_vectors[:2]
coordinates = (features - mean) @ components.T
```

Reject fewer than two features, fewer than two total samples, missing frames,
or inconsistent feature width. Return projection metadata (`mean`, `components`,
and explained-variance ratio) alongside the ordered projected frames. Normalize
only the sign of each component deterministically by requiring its largest
absolute loading to be positive; do not alter frames independently.

**Step 4: Run the focused tests to verify they pass**

Run: `uv run pytest tests/test_representation_visualization.py -k pca -q`

Expected: PASS.

**Step 5: Commit the projection component**

```bash
git add tools/representation_visualization.py tests/test_representation_visualization.py
git commit -m "feat: add globally aligned PCA projection"
```

### Task 4: Render and persist reproducible visualization artifacts

**Files:**
- Modify: `experimental_grow/pyproject.toml`
- Modify: `experimental_grow/tools/representation_visualization.py`
- Modify: `experimental_grow/tests/test_representation_visualization.py`

**Step 1: Add failing artifact tests**

```python
def test_write_visualization_artifacts_writes_gif_snapshots_and_manifest(tmp_path) -> None:
    artifact_paths = write_visualization_artifacts(
        projected_frames=projected_frames,
        projection=projection,
        output_directory=tmp_path,
        title="Hidden layer: layers.0",
        frame_rate=2,
    )

    assert artifact_paths.animation_path.name == "representation_animation.gif"
    assert artifact_paths.animation_path.is_file()
    assert artifact_paths.snapshots_path.is_file()
    assert json.loads(artifact_paths.manifest_path.read_text())["frame_count"] == 2
```

**Step 2: Run the artifact test to verify it fails**

Run: `uv run pytest tests/test_representation_visualization.py -k artifacts -q`

Expected: FAIL because the writer does not exist.

**Step 3: Add the explicit Pillow runtime dependency**

In `pyproject.toml`, add `Pillow` to the project dependencies. Do not add
scikit-learn, Plotly, or an animation framework for version 1.

**Step 4: Implement artifact writing**

Implement `write_visualization_artifacts(...)` to:

- create the output directory with `parents=True, exist_ok=True`;
- calculate `xlim` and `ylim` from the union of all projected coordinates with a
  5% nonzero margin;
- use Matplotlib `FuncAnimation` and `PillowWriter(fps=frame_rate)` to write
  `representation_animation.gif`;
- draw one scatter per label with a stable colormap, plus a title containing
  `step`, `phase`, and available `loss`/`accuracy` values;
- write raw `RepresentationFrame` data and PCA metadata to
  `representation_snapshots.pt` using `torch.save`;
- write a JSON `manifest.json` containing schema version, layer name, probe
  sample count, frame metadata, axis limits, and artifact filenames;
- close the Matplotlib figure in `finally`.

Return a documented `VisualizationArtifactPaths` dataclass. Keep the renderer
independent of `mlflow`, Hydra, and GroMo.

**Step 5: Run the artifact test to verify it passes**

Run: `uv run pytest tests/test_representation_visualization.py -k artifacts -q`

Expected: PASS and no open Matplotlib figures.

**Step 6: Commit renderer and dependency changes**

```bash
git add pyproject.toml tools/representation_visualization.py tests/test_representation_visualization.py
git commit -m "feat: render hidden representation animations"
```

### Task 5: Add an opt-in Hydra visualization interface

**Files:**
- Create: `experimental_grow/hydra_script/configs/visualization/default.yaml`
- Create: `experimental_grow/hydra_script/configs/visualization/enabled.yaml`
- Create: `experimental_grow/hydra_script/configs/model/mlp.yaml`
- Modify: `experimental_grow/hydra_script/configs/config.yaml`
- Modify: `experimental_grow/hydra_script/tests/test_visualization_config.py`

**Step 1: Add tests for the enabled override**

```python
def test_visualization_enabled_override_has_safe_reference_values() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["visualization=enabled", "model=mlp"])

    assert cfg.visualization.enabled is True
    assert cfg.visualization.layer_name == "layers.0"
    assert cfg.visualization.maximum_probe_samples >= 2
```

**Step 2: Run the configuration tests to verify they fail**

Run: `uv run pytest hydra_script/tests/test_visualization_config.py -q`

Expected: FAIL because the default and enabled config files are absent.

**Step 3: Add the configuration files**

Add `visualization: default` to the root config defaults list. Set the default
configuration to:

```yaml
enabled: false
layer_name: ""
maximum_probe_samples: 200
seed: ${general.seed}
capture_every_n_steps: 1
frame_rate: 2
artifact_directory: visualization
```

Set `visualization/enabled.yaml` to override only `enabled: true` and
`layer_name: layers.0`. Add `model/mlp.yaml`, copying the already-supported
model arguments from `models/configs/mlp.yml` so the reference command does not
depend on an out-of-tree config file.

**Step 4: Run the configuration tests to verify they pass**

Run: `uv run pytest hydra_script/tests/test_visualization_config.py -q`

Expected: PASS.

**Step 5: Commit configuration support**

```bash
git add hydra_script/configs hydra_script/tests/test_visualization_config.py
git commit -m "feat: configure hidden representation visualization"
```

### Task 6: Wire the recorder into training and growth boundaries

**Files:**
- Modify: `experimental_grow/hydra_script/train_and_grow.py`
- Modify: `experimental_grow/hydra_script/tests/test_completion_step_logs.py`
- Create: `experimental_grow/hydra_script/tests/test_visualization_lifecycle.py`

**Step 1: Write the failing lifecycle tests around a small fake model and loader**

Test an extracted, pure helper such as `capture_visualization_frame(...)` rather
than making unit tests run MLflow or download a dataset. Cover these boundaries:

```python
def test_growth_step_records_pre_and_post_growth_frames() -> None:
    # A spy recorder receives (step=2, phase="pre_growth") then
    # (step=2, phase="post_growth") around the fake growth callback.
    ...


def test_disabled_visualization_creates_no_recorder_or_artifacts() -> None:
    ...
```

**Step 2: Run the lifecycle tests to verify they fail**

Run: `uv run pytest hydra_script/tests/test_visualization_lifecycle.py -q`

Expected: FAIL because no visualization lifecycle helper exists.

**Step 3: Integrate without modifying GroMo**

In `_run_trainning`:

1. Once `test_loader`, model, and loss/metrics exist, build one `ProbeBatch`
   from `test_loader` and one `RepresentationRecorder` if
   `cfg.visualization.enabled` is true. Use the configured layer name.
2. Capture `step=0, phase="initial"` after initial evaluation.
3. Just before `perform_growth_step`, capture `phase="pre_growth"`; immediately
   after it returns and before optimizer reinitialization, capture
   `phase="post_growth"`.
4. For non-growth steps that satisfy `step % capture_every_n_steps == 0`, capture
   `phase="training"` after metrics are assembled. Do not duplicate the
   post-growth frame at the same step.
5. Pass only scalar `loss` and `accuracy` metrics to the frame metadata. Use
   `test_loss`/`test_accuracy` after evaluation for all normal-step frames.

Put the callback mechanics in a small documented helper so the loop retains its
current control flow. If `layer_name` is invalid or capture fails, raise a clear
`ValueError` before changing growth state; visualization is opt-in, but an
enabled invalid config must not silently omit the artifact.

**Step 4: Run lifecycle tests to verify they pass**

Run: `uv run pytest hydra_script/tests/test_visualization_lifecycle.py hydra_script/tests/test_completion_step_logs.py -q`

Expected: PASS.

**Step 5: Commit the training integration**

```bash
git add hydra_script/train_and_grow.py hydra_script/tests
git commit -m "feat: capture representations during growth training"
```

### Task 7: Finalize artifacts, MLflow logging, documentation, and the CPU smoke test

**Files:**
- Modify: `experimental_grow/hydra_script/train_and_grow.py`
- Modify: `experimental_grow/hydra_script/ReadMe.md`
- Create: `experimental_grow/hydra_script/tests/test_visualization_smoke.py`

**Step 1: Write the failing smoke test**

Use a temporary directory, CPU, a tiny deterministic two-class `TensorDataset`,
and the `model=mlp` configuration. Arrange one scheduled growth operation that
does not change the observed `layers.0` width. Assert that the run produces a
GIF, `.pt` snapshots, and manifest with `initial`, `pre_growth`, and
`post_growth` phases. Stub only MLflow's remote/logging calls; do not stub the
recorder, PCA, renderer, or the growth callback.

**Step 2: Run the smoke test to verify it fails**

Run: `uv run pytest hydra_script/tests/test_visualization_smoke.py -q`

Expected: FAIL because finalization and MLflow artifact logging are not wired.

**Step 3: Finalize at normal training completion and failure-safe boundaries**

At the end of `_run_trainning`, if frames exist:

1. call `GlobalPCAProjector().fit_transform(frames)`;
2. create `<Hydra runtime output directory>/<artifact_directory>`;
3. call `write_visualization_artifacts`;
4. log the three returned paths with `mlflow.log_artifact` under the
   `visualization` artifact path;
5. log `visualization_frame_count` and the two PCA explained-variance ratios;
6. emit the artifact paths through the existing logger.

If a run terminates before two valid PCA dimensions are available, log a clear
warning and raw snapshots/manifest only; do not cause a successful training run
to fail during optional rendering. Capture configuration errors remain hard
failures as specified in Task 6.

Document the run command and the stable-width constraint in
`hydra_script/ReadMe.md`:

```bash
uv run python -m hydra_script.train_and_grow \
  model=mlp visualization=enabled general.device=cpu
```

**Step 4: Run the smoke test to verify it passes**

Run: `uv run pytest hydra_script/tests/test_visualization_smoke.py -q`

Expected: PASS and the temporary directory contains all three artifact types.

**Step 5: Run the relevant test suite and linting**

Run: `uv run pytest tests/test_representation_visualization.py hydra_script/tests/test_visualization_config.py hydra_script/tests/test_visualization_lifecycle.py hydra_script/tests/test_visualization_smoke.py -q`

Expected: PASS.

Run: `uv run ruff check tools/representation_visualization.py hydra_script/train_and_grow.py tests/test_representation_visualization.py hydra_script/tests/test_visualization_config.py hydra_script/tests/test_visualization_lifecycle.py hydra_script/tests/test_visualization_smoke.py`

Expected: PASS.

**Step 6: Commit the complete reference experiment**

```bash
git add hydra_script/train_and_grow.py hydra_script/ReadMe.md hydra_script/tests
git commit -m "feat: export hidden representation visualization artifacts"
```

### Task 8: Verify against the approved scope

**Files:**
- Review: `experimental_grow/tools/representation_visualization.py`
- Review: `experimental_grow/hydra_script/train_and_grow.py`
- Review: `experimental_grow/hydra_script/configs/visualization/`

**Step 1: Check the package boundary**

Run: `git -C /Users/strivaud/Projects/research/gromo diff --stat HEAD~1..HEAD`

Expected: no GroMo source-module changes attributable to this feature; the
implementation resides in `experimental_grow`.

**Step 2: Check forbidden dependencies and concepts**

Run: `rg -n "greedy.learning|langevin|distill|sklearn|umap|TSNE" tools/representation_visualization.py hydra_script/train_and_grow.py pyproject.toml`

Expected: no matches, except intentionally documented future work outside the
implementation files.

**Step 3: Manually inspect the GIF**

Open the smoke-test or a local reference-run artifact. Confirm labels stay
coloured consistently, axes do not move between frames, labels/title metadata
change at checkpoints, and growth boundaries show distinct `pre_growth` and
`post_growth` frames.

**Step 4: Commit any verification-only documentation correction**

```bash
git add hydra_script/ReadMe.md
git commit -m "docs: clarify representation visualization constraints"
```

Only make this commit if the review revealed a documentation correction.
