# Hidden-representation visualization: reference experiment design

## Decision

Build the first visualization integration in `experimental_grow`, not in the
GroMo core package. The milestone is a reproducible Hydra training-and-growth
experiment that renders a two-dimensional animation of one hidden-layer
representation. It validates the capture, temporal alignment, projection, and
artifact interfaces against real GroMo growth events before committing to a
public library API.

The visualization is a diagnostic for learning and growth. It is not part of
the greedy-learning/Langevin demo, and the milestone must not depend on that
code or import it.

## Scope

The reference experiment initially targets an existing MLP or residual MLP
workflow. A configured layer is observed on a fixed, stratified probe set at
selected epoch checkpoints and immediately before and after growth events.
The rendered animation shows the projected probe activations over time.

Each rendered frame includes:

- probe samples coloured by ground-truth class;
- optional indication of the predicted class or prediction error;
- epoch, loss, accuracy, and growth-event metadata;
- stable axes shared by every frame.

The experiment produces a self-contained animation artifact, the captured
snapshots needed to reproduce it, and the resolved Hydra configuration. When
MLflow is enabled, those artifacts are logged with the corresponding run.

## Architecture

The implementation is divided into four small components located in
`experimental_grow`.

1. **Probe-set provider.** Chooses the fixed validation subset once, retaining
   sample identifiers, labels, and deterministic ordering. It must be separate
   from the shuffled training batch stream.
2. **Representation recorder.** Resolves the configured module name, registers
   a PyTorch forward hook for the duration of a probe evaluation, and records a
   detached CPU tensor plus frame metadata. It validates the layer name and
   output shape, and always removes the hook.
3. **Temporal projector.** Takes the ordered snapshots and maps every sample to
   two coordinates. Version 1 uses PCA fitted once over the run's full,
   reference representation matrix. It then applies that single basis to each
   frame. The component retains raw snapshots so later projectors can be added
   without changing capture.
4. **Renderer and artifact writer.** Renders consistently scaled frames and
   encodes an animation. It receives frame data only; it does not inspect
   GroMo models or training state.

The training loop calls the recorder at an explicit checkpoint boundary. A
growth event yields two distinct checkpoints, `pre_growth` and `post_growth`,
so capacity changes are visible even when they happen inside one epoch.

## Temporal projection contract

The first projector is PCA, fitted globally rather than independently for each
frame. Per-frame PCA permits arbitrary sign, axis order, and rotation changes;
that would make apparent motion an artifact of projection instead of learning.

The recorder therefore requires every snapshot to have the same probe-sample
order. Hidden feature widths may change at growth events. For version 1, the
reference workflow must select a layer whose output dimensionality is stable
through the inspected growth events. Supporting a changing feature width is a
deliberate follow-up design problem, likely requiring a common learned probe or
an explicit feature-alignment adapter.

The PCA fit records its mean, components, and explained variance with the
artifact. The renderer determines axis limits from all projected frames, not
one frame at a time.

## Hydra interface

Add a `visualization` configuration group with an explicit disabled default.
The enabled reference configuration specifies the layer path, checkpoint
schedule, maximum probe-set size, class-stratified sampling seed, output
format, and PCA projection settings. It must be possible to run the existing
workflow unchanged when visualization is disabled.

The configuration must select an existing model layer by name, rather than
introducing GroMo-specific visualization methods. This makes the adapter work
with standard `torch.nn.Module` hierarchies and keeps it outside the GroMo
package while the interface matures.

## Error handling and resource limits

Fail early with a clear error if the configured layer cannot be resolved, its
hook output is not a tensor, a probe snapshot changes sample count/order, or
the selected run has no captured frames. Capture tensors under
`torch.inference_mode()` and move them to CPU immediately. Enforce a configured
probe-set maximum and checkpoint cadence to avoid retaining training activations
or producing unbounded artifact sizes.

## Validation

Tests in `experimental_grow` cover layer resolution, hook cleanup, snapshot
shape validation, deterministic probe selection, and PCA alignment across
frames. A small CPU smoke test runs a minimal growing MLP workflow and asserts
that it writes the expected frame data and animation artifact. The smoke test
does not assert scientific accuracy or a visual movement pattern.

## Explicit non-goals

- Porting Langevin, score-field, greedy-learning, or distillation code.
- Bringing visualization dependencies into GroMo core.
- General support for every architecture or feature-width-changing layer.
- Per-frame t-SNE, MDS, or autoencoder fitting.
- Treating a 2D projection as a faithful representation of all high-dimensional
  geometry.

## Follow-up path

Once the reference run reliably produces meaningful artifacts, extract the
recorder/projector/renderer contract into a portable visualization package or
optional dependency. The next projections should be selected only after the
PCA reference exposes a real limitation: supervised linear probe for class
separation, then a temporally aligned nonlinear method if necessary. CNN and
Transformer support follows by defining explicit tensor-reduction policies
before projection.
