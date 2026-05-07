# openpi_zzk BRX Isaac Lab Simulation

This directory is intentionally separate from `scripts/custom`.

Start Isaac Lab first. This process only owns simulation, cameras, qpos, WebRTC
rendering, and HTTP action exchange. It does not import openpi and it does not
own the language prompt.

```bash
cd /home/kemove/zzk_data/IsaacLab
CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p scripts/openpi_zzk/run_brx_openpi_policy.py \
  --headless \
  --livestream 2 \
  --enable_cameras \
  --device cuda:0 \
  --mode remote_policy \
  --policy_server_url http://127.0.0.1:8777/infer \
  --urdf_path /home/kemove/zzk_data/IsaacLab/BRX042501/BRX042501_wheel.urdf \
  --camera_pose_mode link \
  --print_action_summary
```

Then start the policy server from the openpi environment. This process owns the
model checkpoint and prompt:

```bash
cd /home/kemove/zzk_data/openpi
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run examples/brx/brx_policy_http_server.py \
  --config-name pi05_brx_finetune \
  --checkpoint-dir /home/kemove/zzk_data/openpi/checkpoints/pi05_brx_finetune/brx_23d_lora/3000 \
  --host 127.0.0.1 \
  --port 8777 \
  --default-prompt "grab the small blocks pick them up and put them in the bucket"
```

Replay recorded HDF5 actions without loading openpi:

```bash
./isaaclab.sh -p scripts/openpi_zzk/run_brx_openpi_policy.py \
  --mode replay_hdf5 \
  --action_hdf5 /home/kemove/ACT_Datasets/episode_0.hdf5 \
  --urdf_path /home/kemove/zzk_data/IsaacLab/BRX042501/BRX042501_wheel.urdf
```

Both modes command 23D absolute qpos in the BRX order used by `openpi.policies.brx_policy.BRX_JOINT_NAMES`.
