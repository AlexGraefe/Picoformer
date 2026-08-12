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
