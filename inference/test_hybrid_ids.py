import unittest

import numpy as np

from inference import HybridIDS


VALID_FLOW = {
    "destination_port": 443,
    "protocol": 6,
    "packet_count": 16,
    "byte_count": 4326,
    "flow_duration_us": 141385,
}


class HybridIDSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = HybridIDS()

    def test_valid_hybrid_input(self):
        result = self.ids.predict(VALID_FLOW)
        self.assertIn(result["prediction"], self.ids.class_names)
        self.assertEqual(list(result["probabilities"]), self.ids.class_names)
        self.assertTrue(np.isclose(sum(result["probabilities"].values()), 1.0))
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_random_forest_mode(self):
        result = self.ids.predict(VALID_FLOW, model="random_forest")
        self.assertEqual(list(result["probabilities"]), self.ids.class_names)

    def test_missing_feature_is_rejected(self):
        flow = dict(VALID_FLOW)
        del flow["protocol"]
        with self.assertRaisesRegex(ValueError, "Missing required feature"):
            self.ids.predict(flow)

    def test_unexpected_feature_is_rejected(self):
        flow = dict(VALID_FLOW, source_ip="10.0.0.1")
        with self.assertRaisesRegex(ValueError, "Unexpected feature"):
            self.ids.predict(flow)

    def test_invalid_numeric_value_is_rejected(self):
        flow = dict(VALID_FLOW, byte_count="not-a-number")
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            self.ids.predict(flow)


if __name__ == "__main__":
    unittest.main()
