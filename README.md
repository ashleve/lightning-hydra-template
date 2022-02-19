This branch contains example wandb callbacks.

Callbacks which support reproducibility:

- **WatchModel**
- **UploadCodeAsArtifact**
- **UploadCheckpointsAsArtifact**

Callbacks which provide examples of logging custom visualisations:

- **LogConfusionMatrix**
- **LogF1PrecRecHeatmap**
- **LogImagePredictions**

To try all of the callbacks at once, run the following command:

```bash
python train.py logger=wandb callbacks=wandb
```
