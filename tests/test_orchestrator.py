import unittest

from src.downstream.link_adaptation import LinkAdaptationModule
from src.orchestrator.pipeline import SignalOrchestrator


class TestLinkAdaptationModule(unittest.TestCase):
    def setUp(self):
        self.module = LinkAdaptationModule(min_confidence=0.6)
        self.orchestrator = SignalOrchestrator(self.module)

    def test_noise_route_exits_early(self):
        prediction = {
            "predicted_class_name": "noise",
            "snr_estimate": -2.0,
            "confidence": 0.98,
        }
        result = self.orchestrator.route(prediction)
        self.assertEqual(result["route"], "noise_exit")
        self.assertEqual(result["downstream"]["status"], "skipped")

    def test_wifi_low_confidence_defers(self):
        prediction = {
            "predicted_class_name": "wifi",
            "snr_estimate": 22.0,
            "confidence": 0.55,
        }
        result = self.orchestrator.route(prediction)
        self.assertEqual(result["route"], "signal_to_downstream")
        self.assertEqual(result["downstream"]["status"], "deferred")
        self.assertIsNone(result["downstream"]["recommended_mcs"])

    def test_wifi_high_confidence_recommends_mcs(self):
        prediction = {
            "predicted_class_name": "wifi",
            "snr_estimate": 24.5,
            "confidence": 0.93,
        }
        result = self.orchestrator.route(prediction)
        self.assertEqual(result["downstream"]["status"], "ok")
        self.assertEqual(result["downstream"]["recommended_action"], "transmit")
        self.assertEqual(result["downstream"]["recommended_mcs"], "MCS7")


if __name__ == "__main__":
    unittest.main()
