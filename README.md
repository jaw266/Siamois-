# Détection de Changement Siamoise — PFE

Système de détection de changement bi-temporel basé sur un réseau de neurones siamois (Siamese Network), déployé avec ONNX Runtime et connecté à HiveMQ Cloud via MQTT.

## Architecture

```
Images T1 + T2
      │
      ▼
┌─────────────────────┐
│  Modèle Siamois     │  ← ONNX Runtime (CPU)
│  (Change Detector)  │     + TTA (4 augmentations)
└─────────────────────┘
      │
      ▼
  Résultat JSON
  (prediction, probabilité, masque, bbox)
      │
      ▼
┌─────────────────┐
│   HiveMQ Cloud  │  ← MQTT TLS (port 8883)
│   Broker MQTT   │
└─────────────────┘
      │
      ▼
┌─────────────────┐     ┌─────────────┐
│  Dashboard Web  │     │  SQLite DB  │
│  (WebSocket)    │     │  (historique│
└─────────────────┘     └─────────────┘
```

## Fonctionnalités

- **Inférence ONNX** : modèle siamois bi-temporel optimisé pour CPU
- **TTA** (Test-Time Augmentation) : 4 augmentations pour améliorer la robustesse
- **Post-traitement** : suppression des petites régions, remplissage des trous
- **Niveaux de sévérité** : NORMAL / LOW / MEDIUM / HIGH
- **Publication MQTT** : résultats publiés en temps réel sur HiveMQ Cloud
- **Dashboard Web** : interface temps réel via WebSocket MQTT
- **Base de données** : historique des détections dans SQLite

## Structure du projet

```
pfe_hivemq_ready_pack/
├── backend/
│   ├── inference_onnx.py              # Moteur d'inférence ONNX
│   ├── inference_onnx_mqtt_hivemq.py  # Inférence + publication HiveMQ
│   ├── mqtt_config_hivemq.py          # Configuration broker MQTT
│   ├── db_subscriber.py               # Subscriber MQTT → SQLite
│   └── view_db.py                     # Visualisation base de données
├── model/
│   ├── siamese_change_detector_phase2bis.onnx  # Modèle ONNX
│   └── deployment_config_phase2bis.json        # Paramètres de déploiement
├── web/
│   └── index_hivemq.html              # Dashboard temps réel
├── test_images/                       # Images de test bi-temporelles
│   ├── test_01_dataset/  (T1.png, T2.png)
│   ├── test_02_dataset/
│   ├── test_03_tunisie/
│   ├── test_04_tunisie/
│   ├── test_05_dataset/
│   └── test_07_dataset/
├── outputs/                           # Résultats générés
├── run_tunisie.bat                    # Script tests Tunisie
└── requirements.txt                   # Dépendances Python
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Lancer une inférence et publier sur HiveMQ

```bash
cd backend
python inference_onnx_mqtt_hivemq.py \
  --t1 ../test_images/test_07_dataset/T1.png \
  --t2 ../test_images/test_07_dataset/T2.png \
  --id_test test_07_dataset
```

### Enregistrer les résultats dans SQLite

```bash
cd backend
python db_subscriber.py   # laisser tourner en arrière-plan
```

### Visualiser la base de données

```bash
cd backend
python view_db.py
```

### Dashboard Web

Ouvrir `web/index_hivemq.html` dans un navigateur — connexion automatique au broker HiveMQ Cloud.

## Configuration HiveMQ

Éditer `backend/mqtt_config_hivemq.py` :

```python
BROKER   = "<votre-cluster>.s1.eu.hivemq.cloud"
PORT     = 8883
USERNAME = "<username>"
PASSWORD = "<password>"
TOPIC    = "pfe/siamese/anomaly/result"
```

## Format du résultat JSON

```json
{
  "id_test": "test_07_dataset",
  "timestamp": "2026-06-07 17:59:59",
  "prediction": "CHANGE",
  "classification_probability": 0.9996,
  "classification_threshold": 0.61,
  "segmentation_threshold": 0.41,
  "change_area_ratio": 0.0235,
  "severity": "HIGH",
  "bbox": { "x_min": 35, "y_min": 128, "x_max": 204, "y_max": 233 },
  "mask_path": "outputs/masks/test_07_dataset_pred_mask.png",
  "overlay_path": "outputs/overlays/test_07_dataset_overlay.png",
  "tta_enabled": true
}
```

## Dépendances

- `onnxruntime` — inférence du modèle
- `paho-mqtt` — client MQTT
- `Pillow` — traitement d'images
- `numpy` — calcul matriciel
- `scipy` — post-traitement du masque

## Auteur

Projet de Fin d'Études (PFE) — Détection de changement par télédétection satellitaire.
