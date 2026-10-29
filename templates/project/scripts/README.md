# Scripts

Project-specific scripts. **This directory is intentionally unconstrained.**

It exists so one-off automation does not pollute `commands/` or the component
packages. Organize it however the current project needs; the DLEF template does
not prescribe subdirectories.

Guidelines:

- Anything here is project-specific and outside the framework contract.
- If logic becomes reusable by training/eval/sweep, move it into `myproject/`
  (or DLEF core/adapters if it is cross-project) and keep a thin script.
- Scripts do not write run manifests, metrics or checkpoints by hand.
- Generated data belongs under `$DLEF_DATA_ROOT`; generated outputs under
  `outputs/`.

The template's own generic maintenance utilities live in `tools/`, not here.
