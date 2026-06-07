import json
import sqlite3
import ssl
import time
from pathlib import Path

import paho.mqtt.client as mqtt


# =========================
# HiveMQ Cloud parameters
# =========================
BROKER = "e5d69244c7204e68b4a5119be24d9bc8.s1.eu.hivemq.cloud"
PORT = 8883
TOPIC = "pfe/siamese/anomaly/result"

USERNAME = "hivemq.webclient.1780849948851"
PASSWORD = "N3wi&v!jXYe4>A<56yLE"

# =========================
# SQLite database path
# =========================
DB_PATH = Path("../outputs/results.db")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_test TEXT,
            timestamp TEXT,
            image_t1 TEXT,
            image_t2 TEXT,
            prediction TEXT,
            classification_probability REAL,
            classification_threshold REAL,
            segmentation_threshold REAL,
            change_area_ratio REAL,
            severity TEXT,
            bbox TEXT,
            mask_path TEXT,
            overlay_path TEXT,
            raw_json TEXT,
            received_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_result(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO detection_results (
            id_test,
            timestamp,
            image_t1,
            image_t2,
            prediction,
            classification_probability,
            classification_threshold,
            segmentation_threshold,
            change_area_ratio,
            severity,
            bbox,
            mask_path,
            overlay_path,
            raw_json,
            received_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("id_test"),
        data.get("timestamp"),
        data.get("image_t1"),
        data.get("image_t2"),
        data.get("prediction"),
        data.get("classification_probability"),
        data.get("classification_threshold"),
        data.get("segmentation_threshold"),
        data.get("change_area_ratio"),
        data.get("severity"),
        json.dumps(data.get("bbox"), ensure_ascii=False),
        data.get("mask_path"),
        data.get("overlay_path"),
        json.dumps(data, ensure_ascii=False),
        time.strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def on_connect(client, userdata, flags, reason_code, properties=None):
    print("Connecté à HiveMQ Cloud.")
    print("Reason code:", reason_code)
    print("Subscribe topic:", TOPIC)
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    print("\n==============================")
    print("Message reçu depuis HiveMQ")
    print("Topic:", msg.topic)

    try:
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)

        insert_result(data)

        print("Résultat enregistré dans SQLite.")
        print("ID test:", data.get("id_test"))
        print("Prediction:", data.get("prediction"))
        print("Probability:", data.get("classification_probability"))
        print("Severity:", data.get("severity"))
        print("==============================")

    except Exception as e:
        print("Erreur lors du traitement du message:", e)


def main():
    init_db()

    print("Base SQLite prête:", DB_PATH)
    print("Connexion à HiveMQ Cloud...")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    client.username_pw_set(USERNAME, PASSWORD)

    client.tls_set(
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLS_CLIENT
    )

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)

    print("En attente des messages MQTT...")
    client.loop_forever()


if __name__ == "__main__":
    main()