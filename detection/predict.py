mport random

def predict_traffic(packet):
    """
    Simulates intrusion detection.
    Returns either 'Normal' or 'Attack'.
    """

    prediction = random.choice(["Normal", "Attack"])

    print("Packet:", packet)
    print("Prediction:", prediction)

    return prediction


if _name_ == "_main_":
    sample_packet = {
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "protocol": "TCP",
        "packet_size": 512
    }

    predict_traffic(sample_packet)
