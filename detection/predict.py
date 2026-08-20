import joblib
import numpy as np
from datetime import datetime

class ThreatDetector:

    def __init__(self):
        self.model = joblib.load("models/ids_model.pkl")

        self.labels = {
            0: "Normal",
            1: "Brute Force",
            2: "Botnet",
            3: "Web Attack"
        }

    def detect(self, features):

        features = np.array(features).reshape(1, -1)

        prediction = self.model.predict(features)[0]

        probability = max(self.model.predict_proba(features)[0])

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attack": self.labels[prediction],
            "confidence": round(float(probability) * 100, 2),
            "status": "Malicious" if prediction != 0 else "Normal"
        }

        return result


if __name__ == "__main__":

    detector = ThreatDetector()

    sample_flow = [
        150,
        45,
        0,
        300,
        1200,
        10,
        0,
        8,
        250,
        3
    ]

    result = detector.detect(sample_flow)

    print("========== Detection Result ==========")
    print("Time       :", result["timestamp"])
    print("Attack     :", result["attack"])
    print("Confidence :", result["confidence"], "%")
    print("Status     :", result["status"])
