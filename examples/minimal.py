"""Run the mathematical core on tiny synthetic arrays, without a robot."""

import numpy as np

from timelydagger import BridgePCA, FeedbackGuidedThresholdAdaptation


# Replace these arrays with frozen bridge features from your VLA.
id_features = np.array([[-2.0, 0.00], [-1.0, 0.01], [0.0, -0.01], [1.0, 0.02]])
id_calibration_rollouts = [
    np.array([[0.0, 0.01], [0.5, 0.03]]),
    np.array([[0.0, -0.02], [0.5, 0.04]]),
    np.array([[0.0, 0.01], [0.5, 0.05]]),
]

monitor = BridgePCA(rank=1).fit(id_features)
initial_threshold = monitor.calibrate(id_calibration_rollouts, alpha=0.05)
fta = FeedbackGuidedThresholdAdaptation(initial_threshold, block_length=2)

bridge_feature = np.array([0.5, 0.15])
print("Bridge-PCA score:", monitor.score(bridge_feature))
print("Request expert:", monitor.requests_help(bridge_feature, fta.threshold))

# Once an expert intervention completes, compare the preceding robot motion
# with the initial expert motion. Policy predictions are evaluated at the
# expert's observations but are NOT executed during expert control.
feedback = fta.observe_intervention(
    previous_robot_motion=np.array([1.0, 0.0]),
    initial_expert_motion=np.array([-1.0, 0.0]),
    policy_actions=np.zeros((2, 2)),
    expert_actions=np.array([[-0.3, 0.0], [-0.2, 0.0]]),
)
print("Timing cue:", feedback.cue)
print("Threshold for subsequent episodes:", fta.threshold)
