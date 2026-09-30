<div align="center">

# TimelyDAgger

### Timing-Aware Expert Querying for VLA Policy Improvement

Zhixuan Zhao<sup>1,6,†</sup> · Peiyan Li<sup>1,6,†</sup> · Enhao Zhang<sup>2,6</sup> · Yueran Tao<sup>1,6</sup><br>
Hao Wang<sup>3,6</sup> · Chenghao Yue<sup>1,6</sup> · Lei Lv<sup>4,6</sup> · Wentao Zhao<sup>1,6</sup> · Jiahao Chen<sup>5,6</sup><br>
Xin Liu<sup>1,6</sup> · Kangyao Huang<sup>1,6</sup> · Yu Luo<sup>1,6,*</sup> · Huaping Liu<sup>1,6,*</sup>

<sub><sup>1</sup>Tsinghua University &nbsp; <sup>2</sup>Imperial College London &nbsp; <sup>3</sup>Dalian University of Technology<br>
<sup>4</sup>Tongji University &nbsp; <sup>5</sup>Peking University &nbsp; <sup>6</sup>SEEN·E Robotics</sub><br>
<sub>† Equal contribution &nbsp;&nbsp; * Corresponding authors</sub>

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2609.33157-B31B1B?style=flat-square)](https://arxiv.org/abs/2609.33157)
[![Project page](https://img.shields.io/badge/Project-Page-2C817A?style=flat-square)](https://seen-e.github.io/TimelyDagger/)
[![Code website](https://img.shields.io/badge/Code-Website-3B7297?style=flat-square)](https://hrinnnn.github.io/TimelyDAgger-Code/)
![Python](https://img.shields.io/badge/Python-3.10%2B-4776A7?style=flat-square)
[![License](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)](LICENSE)

</div>

**TimelyDAgger** studies when a robot should ask an expert to take over. The timing of that request determines where the expert demonstration begins and, in turn, what the policy can learn from the collected data. The method combines **Bridge-PCA** for monitoring internal VLA features with **Feedback-guided Threshold Adaptation (FTA)** to refine the timing of future requests.

<div align="center">
  <a href="https://seen-e.github.io/TimelyDagger/">
    <img src="https://seen-e.github.io/TimelyDagger/assets/method_overview.jpg" alt="TimelyDAgger: Bridge-PCA monitoring, online expert intervention and threshold adaptation, followed by policy update" width="100%">
  </a>
  <sub>Method overview. See the <a href="https://seen-e.github.io/TimelyDagger/">project page</a> for the paper, figures, and robot videos.</sub>
</div>

## News

- **2026-10-01:** Bridge-PCA and FTA method code released.

## Getting started

The code uses Python 3.10+ and NumPy. Clone the repository and run the included example:

```bash
git clone https://github.com/hrinnnn/TimelyDAgger-Code.git
cd TimelyDAgger-Code
python -m pip install -r requirements.txt
python -m examples.minimal
```

The example uses small synthetic arrays to show the API. It does not require a robot or model checkpoint.

## Method

### Bridge-PCA: decide when to request help

Fit a rank-$r$ principal subspace to bridge features from in-distribution (ID) demonstrations. For a new bridge feature $z_t$, the monitor computes the reconstruction residual

$$
s_{\mathrm{BPCA}}(z_t)=\left\|(I-VV^\top)(z_t-\mu)\right\|_2.
$$

The initial threshold is the $(1-\alpha)$-quantile of **per-rollout maximum scores** on separate successful ID calibration rollouts. The robot requests help when the current score **strictly exceeds** that threshold.

```python
from timelydagger import BridgePCA

monitor = BridgePCA(rank=16).fit(id_bridge_features)  # shape: (N, d)
initial_threshold = monitor.calibrate(id_calibration_rollouts, alpha=0.05)
request_help = monitor.requests_help(current_bridge_feature)
```

The caller extracts bridge features from a frozen VLA and uses the same feature definition for fitting, calibration, and online scoring. Choose the PCA rank before evaluation.

### FTA: refine when future requests occur

After expert takeover, FTA uses the expert's initial actions as timing feedback. It detects a corrective reversal by comparing motion immediately before and after takeover; it also checks whether frozen-policy predictions agree with expert actions at the same observations. A reversal takes priority over agreement. The resulting cue $y_i\in\{+1,0,-1\}$ updates one global threshold:

$$
\gamma_{i+1}=\gamma_i\exp(-\beta y_i).
$$

```python
from timelydagger import FeedbackGuidedThresholdAdaptation

fta = FeedbackGuidedThresholdAdaptation(initial_threshold, block_length=8)

# Call after a completed intervention. Policy actions are predictions at the
# expert's observations; only the expert actions are executed.
feedback = fta.observe_intervention(
    previous_robot_motion,
    initial_expert_motion,
    policy_predicted_actions,
    expert_executed_actions,
)

# Use the updated threshold for subsequent policy queries and episodes.
request_help = monitor.requests_help(current_bridge_feature, fta.threshold)
```

The default $\beta=0.05$ changes the threshold by about 5% for a nonzero cue. This value and the action/motion tolerances in `fta.py` are starting settings for adapting the code to a new robot; **they are not the parameter values validated in the paper's experiments**. Pass motion vectors in one coordinate frame and action vectors in compatible units.

## Code layout

```text
timelydagger/
  bridge_pca.py    # ID subspace, reconstruction score, threshold calibration
  fta.py           # timing feedback and global threshold update
examples/
  minimal.py       # runnable numerical example
tests/
  test_core.py      # checks for the method equations and cue priority
```

This release exposes the Bridge-PCA and FTA components. It accepts features and intervention data from an existing policy/robot interface; the paper and [project page](https://seen-e.github.io/TimelyDagger/) describe the complete data collection and policy learning workflow.

## Citation

If TimelyDAgger is useful in your work, please cite the paper:

```bibtex
@misc{zhao2026timelydagger,
  title = {TimelyDAgger: Timing-Aware Expert Querying for VLA Policy Improvement},
  author = {Zhixuan Zhao and Peiyan Li and Enhao Zhang and Yueran Tao and Hao Wang and Chenghao Yue and Lei Lv and Wentao Zhao and Jiahao Chen and Xin Liu and Kangyao Huang and Yu Luo and Huaping Liu},
  year = {2026},
  eprint = {2609.33157},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2609.33157}
}
```

## License

The code is released under the [MIT License](LICENSE).
