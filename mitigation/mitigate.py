blocked_ips = []

def block_ip(ip):
    if ip not in blocked_ips:
        blocked_ips.append(ip)
        print(f"IP {ip} has been blocked.")
    else:
        print(f"IP {ip} is already blocked.")

def mitigation(prediction, ip):
    """
    Block the IP if an attack is detected.
    """
    if prediction == "Attack":
        print("⚠️ Attack Detected!")
        block_ip(ip)
    else:
        print("✅ Normal Traffic. No action required.")

if _name_ == "_main_":
    prediction = "Attack"
    ip = "10.0.0.2"

    mitigation(prediction, ip)

    print("\nBlocked IPs:")
    print(blocked_ips)
