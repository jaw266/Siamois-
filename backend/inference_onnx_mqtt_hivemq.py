import argparse
import json
import ssl
import sys
import time
import threading

import paho.mqtt.client as mqtt

from inference_onnx import run_inference
from mqtt_config_hivemq import BROKER, PORT, TOPIC, USERNAME, PASSWORD, USE_TLS


def publish_result(result):
    connected = threading.Event()
    error_msg = []

    def on_connect(c, u, f, reason_code, props=None):
        code = str(reason_code)
        if code in ("0", "Success"):
            connected.set()
        else:
            error_msg.append(f"Connexion refusée: {reason_code}")
            connected.set()

    def on_disconnect(c, u, f, reason_code=None, props=None):
        pass

    client_id = f"pfe-inference-{int(time.time())}"
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except Exception:
        client = mqtt.Client(client_id=client_id)

    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect

    if USE_TLS:
        client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
        client.tls_insecure_set(False)

    try:
        client.connect(BROKER, PORT, keepalive=60)
    except Exception as e:
        raise RuntimeError(f"Impossible de joindre le broker: {e}")

    client.loop_start()

    if not connected.wait(timeout=10):
        client.loop_stop()
        raise RuntimeError("Timeout: broker injoignable (vérifier host/port).")

    if error_msg:
        client.loop_stop()
        raise RuntimeError(error_msg[0] + " — vérifier username/password dans mqtt_config_hivemq.py")

    payload = json.dumps(result, ensure_ascii=False)
    info = client.publish(TOPIC, payload=payload, qos=1, retain=True)
    info.wait_for_publish(timeout=10)

    client.loop_stop()
    client.disconnect()
    print(f"Message publié vers HiveMQ Cloud sur topic: {TOPIC}")


def main():
    parser = argparse.ArgumentParser(description="ONNX inference + HiveMQ Cloud MQTT publish")
    parser.add_argument("--model",      default="../model/siamese_change_detector_phase2bis.onnx")
    parser.add_argument("--config",     default="../model/deployment_config_phase2bis.json")
    parser.add_argument("--t1",         required=True)
    parser.add_argument("--t2",         required=True)
    parser.add_argument("--id_test",    default="test_01")
    parser.add_argument("--output_dir", default="../outputs")
    args = parser.parse_args()

    result = run_inference(args.model, args.config, args.t1, args.t2, args.id_test, args.output_dir)
    print("Résultat inference:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("Publication MQTT vers HiveMQ Cloud...")
    publish_result(result)
    print("Terminé.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERREUR:", e, file=sys.stderr)
        sys.exit(1)
