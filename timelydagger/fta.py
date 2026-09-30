"""Global feedback-guided threshold adaptation (FTA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class TimingFeedback:
    """A completed intervention's heuristic timing cue.

    cue: +1 = request earlier, -1 = request later, 0 = keep timing.
    """

    cue: int
    corrective_reversal: bool
    policy_expert_agreement: bool


class FeedbackGuidedThresholdAdaptation:
    """Apply one global threshold update after each completed intervention.

    All motion and action arrays must use consistent physical/normalized units.
    The numerical cue tolerances below are illustrative defaults. They are not
    the paper's measured settings and should be selected for the target robot.
    """

    def __init__(
        self,
        initial_threshold: float,
        block_length: int,
        *,
        beta: float = 0.05,
        agreement_tolerance: float = 0.05,
        reversal_cosine_threshold: float = -0.5,
        motion_epsilon: float = 1e-6,
    ) -> None:
        if initial_threshold <= 0 or block_length < 1 or beta <= 0:
            raise ValueError("threshold, block_length, and beta must be positive")
        if agreement_tolerance < 0 or not -1 <= reversal_cosine_threshold < 0:
            raise ValueError("invalid feedback tolerances")
        self.threshold = float(initial_threshold)
        self.block_length = block_length
        self.beta = beta
        self.agreement_tolerance = agreement_tolerance
        self.reversal_cosine_threshold = reversal_cosine_threshold
        self.motion_epsilon = motion_epsilon

    def timing_feedback(
        self,
        previous_robot_motion: ArrayLike,
        initial_expert_motion: ArrayLike,
        policy_actions: ArrayLike,
        expert_actions: ArrayLike,
    ) -> TimingFeedback:
        """Compare motion around takeover and actions during expert control.

        Motion vectors summarize the robot's last execution block and the
        expert's initial block. Policy actions are predictions at the *same*
        observations as the executed expert actions. The caller supplies the
        first ``block_length`` such paired predictions/actions; policy actions
        must not be executed during takeover.
        """
        before = np.asarray(previous_robot_motion, dtype=np.float64)
        after = np.asarray(initial_expert_motion, dtype=np.float64)
        policy = np.asarray(policy_actions, dtype=np.float64)
        expert = np.asarray(expert_actions, dtype=np.float64)
        if before.ndim != 1 or before.shape != after.shape:
            raise ValueError("motion vectors must have the same 1-D shape")
        if policy.shape != expert.shape or policy.ndim != 2:
            raise ValueError("action blocks must have matching (Delta, action_dim) shapes")
        if policy.shape[0] != self.block_length:
            raise ValueError("provide exactly block_length action pairs")
        if not all(np.isfinite(x).all() for x in (before, after, policy, expert)):
            raise ValueError("feedback inputs must be finite")

        before_norm = float(np.linalg.norm(before))
        after_norm = float(np.linalg.norm(after))
        reversal = False
        if before_norm > self.motion_epsilon and after_norm > self.motion_epsilon:
            cosine = float(np.dot(before, after) / (before_norm * after_norm))
            reversal = cosine <= self.reversal_cosine_threshold

        differences = np.linalg.norm(policy - expert, axis=1)
        agreement = bool(np.all(differences <= self.agreement_tolerance))
        cue = 1 if reversal else -1 if agreement else 0  # correction has priority
        return TimingFeedback(cue, reversal, agreement)

    def update(self, feedback: TimingFeedback) -> float:
        """Apply gamma <- gamma * exp(-beta * cue) globally."""
        self.threshold *= float(np.exp(-self.beta * feedback.cue))
        return self.threshold

    def observe_intervention(
        self,
        previous_robot_motion: ArrayLike,
        initial_expert_motion: ArrayLike,
        policy_actions: ArrayLike,
        expert_actions: ArrayLike,
    ) -> TimingFeedback:
        """Infer feedback and update the threshold after takeover completes."""
        feedback = self.timing_feedback(
            previous_robot_motion, initial_expert_motion, policy_actions, expert_actions
        )
        self.update(feedback)
        return feedback
