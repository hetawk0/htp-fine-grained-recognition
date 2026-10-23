# Archived experiment configs

These YAML files capture historical trials and research matrices. They are not
the recommended starting point for new runs. Use:

- `../soefbanet_resnet50_ddnorm.yaml` for the best recorded single-model clinical run
  (SOE-FBANet full CE; see the repository's `EXPERIMENT_RESULTS.md`).
- `../dd_train.yaml` as the clinical dataset baseline.
- `../default.yaml` or `../smoke.yaml` for synthetic-data setup and smoke tests.

Historical configs are retained for reproducibility. Their `outputs/` paths
refer to local checkpoints that are not included in the repository. Clinical
data and split manifests are also not included.
