import unittest

from cloud_ready_ml_api.app import load_model, predict


class InferenceApiTest(unittest.TestCase):
    def test_predict_returns_known_label(self):
        model = load_model()
        result = predict([0.82, 0.15, 0.12, 0.42], model)
        self.assertIn(result["label"], model["labels"])
        self.assertGreater(result["score"], 0)
        self.assertLessEqual(result["score"], 1)

    def test_feature_length_validation(self):
        model = load_model()
        with self.assertRaises(ValueError):
            predict([0.1, 0.2], model)


if __name__ == "__main__":
    unittest.main()

