"""Small numerical checks of the public method equations; no experiments."""

import unittest

import numpy as np

from timelydagger import BridgePCA, FeedbackGuidedThresholdAdaptation


class BridgePCATests(unittest.TestCase):
    def test_residual_and_episode_maximum_calibration(self) -> None:
        monitor = BridgePCA(rank=1).fit([[-2, 0], [0, 0], [2, 0], [4, 0]])
        self.assertAlmostEqual(monitor.score([1, 3]), 3.0)
        threshold = monitor.calibrate(
            [np.array([[0, 0], [0, y]]) for y in (0.2, 0.4, 0.6, 1.0)],
            alpha=0.25,
        )
        self.assertAlmostEqual(threshold, 1.0)
        self.assertFalse(monitor.requests_help([0, 1.0]))  # strict >
        self.assertTrue(monitor.requests_help([0, 1.1]))


class FTATests(unittest.TestCase):
    def test_reversal_takes_priority_and_lowers_global_threshold(self) -> None:
        fta = FeedbackGuidedThresholdAdaptation(10.0, block_length=2)
        matched = np.zeros((2, 2))
        feedback = fta.observe_intervention([1, 0], [-1, 0], matched, matched)
        self.assertTrue(feedback.corrective_reversal)
        self.assertTrue(feedback.policy_expert_agreement)
        self.assertEqual(feedback.cue, +1)
        self.assertAlmostEqual(fta.threshold, 10 * np.exp(-0.05))

    def test_agreement_postpones_and_disagreement_keeps(self) -> None:
        fta = FeedbackGuidedThresholdAdaptation(10.0, block_length=2)
        zeros = np.zeros((2, 2))
        feedback = fta.observe_intervention([1, 0], [1, 0], zeros, zeros)
        self.assertEqual(feedback.cue, -1)
        self.assertAlmostEqual(fta.threshold, 10 * np.exp(0.05))
        unchanged = fta.threshold
        feedback = fta.observe_intervention([1, 0], [0, 1], zeros, np.ones((2, 2)))
        self.assertEqual(feedback.cue, 0)
        self.assertEqual(fta.threshold, unchanged)


if __name__ == "__main__":
    unittest.main()
