# Picoformer

## Learning-rate and batch-size sweep

The sweep grid is defined in `config/sweep.yaml`. By default it runs all six
combinations of three learning rates and two local batch sizes, sequentially:

```bash
uv run picoformer-sweep --multirun
```

Hydra writes every resolved training configuration to that job's
`outputs/sweeps/.../automodel.yaml`, then launches it with two AutoModel
processes. Each job also gets a separate checkpoint directory.

Edit `hydra.sweeper.params` in `config/sweep.yaml` to make the grid persistent,
or override it for a one-off sweep:

```bash
uv run picoformer-sweep --multirun \
  optimizer.lr=5e-5,1e-4 \
  step_scheduler.local_batch_size=1,2,4
```

AdamW is used by default. Select Muon for a run, or include both optimizers in
the grid, with Hydra's `sweep_optimizer` choice:

```bash
uv run picoformer-sweep --multirun sweep_optimizer=muon
uv run picoformer-sweep --multirun sweep_optimizer=adamw,muon
```

The Muon choice uses Lion for embeddings, normalization weights, and other
non-matrix parameters; matrix parameters use Muon.

To inspect the generated YAML files without starting training, add
`sweep.dry_run=true`.

## Compute-aware hyperparameter scaling laws

`picoformer-scaling` runs a separate Optuna grid-search study for each model-size
and FLOP-target pair in `config/scaling.yaml`. The model sizes and FLOP targets
form a Cartesian sweep, and tokens are derived with `D = C / (6N)`. Every trial uses Muon and
searches the configured learning-rate and global-batch-size candidates while
keeping weight decay fixed. `n_trials` caps the number of grid points evaluated
per study. The objective is the final loss on the independent `validation-*.bin`
split, not training loss.

```bash
uv run picoformer-scaling --config config/scaling.yaml
```

To regenerate the tables, fitted laws, and plot from the existing study
databases without running any new training trials:

```bash
uv run picoformer-analyze-scaling --config config/scaling.yaml
```

Each study is resumable through its SQLite database. The output directory gets:

- `best_settings.csv`: exact best trial plus the geometric median of all trials
  within 0.25% of its validation loss;
- `best_overall.json`: the model size, token count, and Muon hyperparameters
  that achieved the lowest validation loss across all scale points;
- `power_laws.json`: coefficients, exponents, and log-space R-squared for
  `h(C) = a C^b`, fitted independently for every model size using `C = 6 N D`;
- `power_laws.png`: learning-rate and batch-size log-log fits,
  with a separate scaling law for every model size;
- per-trial generated configs, logs, checkpoints, and validation metrics.

The `num_parameters` values in `config/scaling.yaml` are the explicit `N` used
to derive token counts. Keep them synchronized with the exact trainable count
when changing the associated architecture. At least two FLOP-target points are
required for each model size; four or more spanning multiple orders of magnitude
are recommended.
