# 🚀 PPO-Clip Quick Start Guide

All fixes applied. Ready to run PPO training!

---

## Run PPO Experiment Now

### Single Run (Recommended for Testing)
```bash
python3 run_ppo_exp.py
```
- ✅ Runs ~1-5 minutes depending on config
- ✅ Updates every 20 steps (was 300)
- ✅ Saves actor/critic checkpoints
- ✅ Logs to `records/one_run/PPO_<timestamp>/`

**Expected output:**
```
Starting PPO experiment with config: ...
SUMO GUI started via TraCI.
[PPO UPDATE] time=20, batch_size=345
[PPO SAVED] checkpoint saved: ckpt_20
[PPO UPDATE] time=40, batch_size=298
[PPO SAVED] checkpoint saved: ckpt_40
... (more updates)
[PPO FINAL UPDATE] time=289, batch_size=145
[PPO SAVED] final checkpoint saved: final_ckpt_289
PPO experiment finished
```

### Batch Run (Multiple Experiments)
```bash
# Edit runexp.py: ALGORITHM = "PPO"
python3 runexp.py
```
- Queues PPO experiments for all traffic patterns
- Each saves its own checkpoints and logs

### Switch to DQN (Original)
```bash
# Edit runexp.py: ALGORITHM = "DQN"
python3 runexp.py
```

---

## Monitor Results

### View Checkpoints
```bash
ls -lh model/one_run/PPO_*/ckpt_*.h5
```

### Check Training Logs
```bash
tail -50 records/one_run/PPO_*/memories.txt
```

### See Reward Progress
```bash
tail -50 records/one_run/PPO_*/log_rewards.txt
```

### Real-time Monitoring
```bash
python3 run_ppo_exp.py 2>&1 | grep PPO
```

---

## Configuration Tweaks

### Make Training Longer
Edit `conf/one_run/exp.conf`:
```json
"RUN_COUNTS": 5000  // increase from 72000 for testing, or use 72000 for full training
```

### Make Updates More/Less Frequent
Edit `conf/one_run/deeplight_agent.conf`:
```json
"UPDATE_PERIOD": 10,   // more frequent updates (less stable but faster learning)
"UPDATE_PERIOD": 50,   // less frequent updates (more stable but slower learning)
```

### Adjust PPO Hyperparameters
Edit same file:
```json
"CLIP_EPS": 0.1,       // clip ratio (0.1-0.3 recommended)
"PPO_EPOCHS": 5,       // training epochs per batch (default 10)
"PPO_BATCH_SIZE": 32,  // minibatch size (default 64)
"ENTROPY_COEF": 0.02,  // entropy regularization (default 0.01)
```

---

## What's Different from Before

| Before | After |
|--------|-------|
| UPDATE_PERIOD: 300 | UPDATE_PERIOD: 20 |
| No checkpoints saved | Actor/Critic saved after each update |
| Generic logging | `[PPO UPDATE]` / `[PPO SAVED]` tags |
| DQN-only in runexp.py | PPO/DQN switchable with `ALGORITHM` |
| No static main() | `@staticmethod` added for integration |

---

## Files Changed

1. ✅ `conf/one_run/deeplight_agent.conf` – UPDATE_PERIOD: 300 → 20
2. ✅ `traffic_light_ppo.py` – Added checkpointing + logging
3. ✅ `runexp.py` – Added ALGORITHM selector + PPO integration
4. ✅ `run_ppo_exp.py` – Ready to use (no changes needed)
5. ✅ `ppo_agent.py` – No changes (already working)

---

## Troubleshooting

### Issue: "No module named traffic_light_ppo"
```bash
python3 -c "import traffic_light_ppo; print('OK')"
```
If this fails, check `traffic_light_ppo.py` exists in project root.

### Issue: "SUMO not found"
```bash
which sumo
which sumo-gui
```
If missing, install SUMO or set `SUMO_HOME` environment variable.

### Issue: "No checkpoints saved"
- Check `model/one_run/PPO_*/` directory exists
- Verify UPDATE_PERIOD is reached (run for >20 steps)
- Check `[PPO SAVED]` messages in output

### Issue: "Training seems stuck"
- SUMO simulation can be slow; wait 2-3 minutes before stopping
- Check SUMO GUI window if opened (sometimes minimized)
- Try shorter RUN_COUNTS in exp.conf

---

## Performance Tips

1. **Disable SUMO GUI for speed:**
   - Change `run_ppo_exp.py`: `sumoCmd = ["sumo-gui", ...` to `sumoCmd = ["sumo", ...`
   - Or directly use `python3 runexp.py` (already uses `sumo` without GUI)

2. **Parallel runs** (if hardware allows):
   - Run multiple terminal instances with different SEED values
   - Modify `run_ppo_exp.py`: `SEED = <unique_value>`

3. **Monitor GPU usage:**
   ```bash
   nvidia-smi -l 1  # Updates every 1 sec (if GPU available)
   ```

---

## Next: Resuming from Checkpoint

(Feature for future enhancement)

To resume training from a saved checkpoint, would need to:
1. Load actor/critic weights: `agent.actor.load_weights("ckpt_20_actor.h5")`
2. Load episode memory from log files
3. Continue training with same configs

---

## Success Metrics

After running, you should see:

✅ `records/one_run/PPO_<timestamp>/memories.txt` – Training trajectory  
✅ `records/one_run/PPO_<timestamp>/log_rewards.txt` – Detailed rewards  
✅ `model/one_run/PPO_<timestamp>/ckpt_*.h5` – Actor/Critic checkpoints  
✅ `[PPO UPDATE]` and `[PPO SAVED]` messages in console  

If all present → **Training successful!**