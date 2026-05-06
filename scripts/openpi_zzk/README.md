# openpi_zzk BRX Isaac Lab Simulation

This directory is intentionally separate from `scripts/custom`.

Run a trained BRX pi05 policy in Isaac Lab:

```bash
./isaaclab.sh -p scripts/openpi_zzk/run_brx_openpi_policy.py \
  --openpi_root /home/kemove/openpi_zzk \
  --config_name pi05_brx_finetune \
  --checkpoint_dir /home/kemove/openpi_zzk/checkpoints/pi05_brx_finetune/brx_23d/29999 \
  --urdf_path /home/kemove/zzk_data/IsaacLab/BRX042501/BRX042501_wheel.urdf \
  --prompt "move the object smoothly"
```

Replay recorded HDF5 actions without loading openpi:

```bash
./isaaclab.sh -p scripts/openpi_zzk/run_brx_openpi_policy.py \
  --mode replay_hdf5 \
  --action_hdf5 /home/kemove/ACT_Datasets/episode_0.hdf5 \
  --urdf_path /home/kemove/zzk_data/IsaacLab/BRX042501/BRX042501_wheel.urdf
```

Both modes command 23D absolute qpos in the BRX order used by `openpi.policies.brx_policy.BRX_JOINT_NAMES`.
