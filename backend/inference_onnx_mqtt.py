import argparse
import json
import ssl

import paho.mqtt.client as mqtt

from inference_onnx import run_inference


BROKER = "e5d69244c7204e68b4a5119be24d9bc8.s1.eu.hivemq.cloud"
PORT = 8883
TOPIC = "pfe/siamese/anomaly/result"

USERNAME = "hivemq.webclient.1780849948851"
PASSWORD = "N3wi&v!jXYe4>A<56yLE"


def publish_result(result):
    client = mqtt.Client()

    client.username_pw_set(USERNAME, PASSWORD)

    client.tls_set(
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLS_CLIENT
    )

    client.connect(BROKER, PORT, 60)
    client.loop_start()

    payload = json.dumps(result, ensure_ascii=False)
    publish_info = client.publish(TOPIC, payload, qos=0)

    publish_info.wait_for_publish()

    client.loop_stop()
    client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="ONNX inference + HiveMQ MQTT publish")
    parser.add_argument("--model", default="../model/siamese_change_detector_phase2bis.onnx")
    parser.add_argument("--config", default="../model/deployment_config_phase2bis.json")
    parser.add_argument("--t1", required=True)
    parser.add_argument("--t2", required=True)
    parser.add_argument("--id_test", default="test_01")
    parser.add_argument("--output_dir", default="../outputs")
    args = parser.parse_args()

    result = run_inference(
        args.model,
        args.config,
        args.t1,
        args.t2,
        args.id_test,
        args.output_dir
    )

    print("Résultat inference:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("Publication MQTT vers HiveMQ Cloud...")
    publish_result(result)

    print(f"Message publié sur topic: {TOPIC}")


if __name__ == "__main__":
    main()