# TimelyDAgger — minimal method code

Small, dependency-light reference implementation of **Bridge-PCA** and **Feedback-guided Threshold Adaptation (FTA)** from [TimelyDAgger: Timing-Aware Expert Querying for VLA Policy Improvement](https://arxiv.org/abs/2609.33157).

**[Method code website](https://hrinnnn.github.io/TimelyDAgger-Code/)** · **[Research project page](https://seen-e.github.io/TimelyDagger/)**

This repository contains the equations and online threshold update only. It has no model weights, datasets, simulator, robot driver, experiment configuration, benchmark results, or training pipeline. You supply bridge features from a frozen VLA and action/motion pairs from your own intervention interface.

## Install and run

Python 3.10+ and NumPy 1.24+:

```bash
python -m pip install -r requirements.txt
python -m examples.minimal
python -m unittest discover -s tests
```

Run these commands from the repository root. No hardware or data download is needed.

## Bridge-PCA

Given one bridge feature \(z_i\in\mathbb R^d\) per in-distribution (ID) observation, fit a mean \(\mu\) and rank-\(r\) principal subspace with orthonormal basis \(V\). At a policy query, compute:

\[
s_{\mathrm{BPCA}}(z)=\left\|(I-VV^\top)(z-\mu)\right\|_2.
\]

For **separate successful ID calibration rollouts**, take the maximum score in each rollout and use their empirical \((1-\alpha)\)-quantile as the initial threshold. The implementation chooses the upper empirical observation (`method="higher"`) to avoid interpolation between maxima. Request expert help when `score > threshold`.

```python
from timelydagger import BridgePCA

monitor = BridgePCA(rank=16).fit(id_bridge_features)  # shape: (N, d)
gamma = monitor.calibrate(id_calibration_rollouts, alpha=0.05)
request_help = monitor.requests_help(current_bridge_feature)
```

Choose `rank` below both \(N\) and \(d\) before evaluation. `alpha=0.05` is an illustrative default; it is not a formal false-alarm guarantee. The features supplied to `fit`, `calibrate`, and `score` must come from the same frozen feature extraction path. Bridge token pooling and VLA inference are outside this repository.

## FTA

After a completed expert takeover, compare the robot's motion immediately before takeover with the expert's initial motion afterward. Also compare frozen-policy *predictions* and executed expert actions at the same observations for the first \(\Delta\) expert actions. The policy predictions are not executed.

The heuristic timing cue is `+1` for corrective reversal, `-1` for policy–expert agreement without reversal, and `0` otherwise. Corrective reversal takes priority if both signals occur. Apply the **global**, task-level update:

\[
\gamma_{i+1}=\gamma_i\exp(-\beta y_i).
\]

```python
from timelydagger import FeedbackGuidedThresholdAdaptation

fta = FeedbackGuidedThresholdAdaptation(gamma, block_length=8, beta=0.05)
feedback = fta.observe_intervention(
    previous_robot_motion, initial_expert_motion,
    policy_predicted_actions, expert_executed_actions,
)
gamma = fta.threshold  # use for every state in subsequent episodes
```

The default `beta=0.05` changes the threshold by about 5% per nonzero cue and is deliberately conservative. This and the default cue tolerances (`agreement_tolerance=0.05`, `reversal_cosine_threshold=-0.5`) are **illustrative implementation choices**, not values established by the paper's experiments. Use compatible action units, normalize or calibrate tolerances for your robot, and supply exactly `block_length` paired actions. `previous_robot_motion` and `initial_expert_motion` are displacement vectors in the same coordinate frame; the code uses cosine similarity to identify substantial reversal.

FTA uses completed interventions to adjust future request timing. The higher-level rollout loop, expert suffix storage, success filtering, fine-tuning, and evaluation are application-specific and intentionally absent.

## API at a glance

| Component | Inputs | Output |
|---|---|---|
| `BridgePCA.fit` | ID bridge features `(N, d)` | fitted ID subspace |
| `BridgePCA.calibrate` | separate ID rollouts of `(T_i, d)` features | initial threshold |
| `BridgePCA.score` | one bridge feature `(d,)` | reconstruction residual |
| `BridgePCA.requests_help` | feature, optional current threshold | strict threshold decision |
| `FTA.timing_feedback` | before/after motion and paired action blocks | cue `+1`, `0`, or `-1` |
| `FTA.observe_intervention` | same inputs | cue and updated global threshold |

## Citation

If you use this implementation, cite the [TimelyDAgger paper](https://arxiv.org/abs/2609.33157). The source code is released under the [MIT license](LICENSE).
