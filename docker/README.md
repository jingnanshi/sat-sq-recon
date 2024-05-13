Dokcer commands:
```bash
docker run --rm --gpus all -it --entrypoint bash -v /home/jnshi/code/sat-sq-recon:/app/sat-sq-recon -v /mnt/jnshi_data/datasets/hydra_objects_data/spe3r:/app/data/spe3r spe3r:latest
```