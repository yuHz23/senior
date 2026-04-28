"""Start the MQTT-based anomaly detection gateway.

Requires a running MQTT broker (e.g., Mosquitto).
  Install:  https://mosquitto.org/download/
  Start:    mosquitto -v

Usage:
    python scripts/run_mqtt_gateway.py --dataset ciciot
    python scripts/run_mqtt_gateway.py --dataset ciciot --broker 192.168.1.10
    python scripts/run_mqtt_gateway.py --dataset intel  --broker localhost --port 1883

Publish test data (in another terminal):
    mosquitto_pub -t iot/gateway/ciciot/data \
        -m '{"features": [0.1, 0.2, 0.3, 0.4, 0.5], "device_id": "router_01"}'

Subscribe to alerts:
    mosquitto_sub -t iot/gateway/ciciot/alerts
"""
import argparse
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.inference.detector import AnomalyDetector
from src.inference.mqtt_gateway import MQTTGateway
from src.data.stream_simulator import CICIOT_FEATURES, INTEL_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  choices=["ciciot", "intel"], required=True)
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--broker",   default="localhost", help="MQTT broker host")
    parser.add_argument("--port",     type=int, default=1883)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--device",   default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg     = cfg["data"][args.dataset]
    mcfg     = cfg["model"]
    features = CICIOT_FEATURES if args.dataset == "ciciot" else INTEL_FEATURES

    detector = AnomalyDetector(
        model_path=cfg["saved_models"][f"{args.dataset}_model"],
        scaler_path=dcfg["scaler_path"],
        threshold_path=cfg["saved_models"][f"{args.dataset}_threshold"],
        input_size=len(features),
        hidden_size=mcfg["hidden_size"],
        bottleneck_size=mcfg["bottleneck_size"],
        seq_len=mcfg["window_size"],
        num_layers=mcfg["num_layers"],
        device=args.device,
    )

    gateway = MQTTGateway(
        detector=detector,
        broker_host=args.broker,
        broker_port=args.port,
        dataset=args.dataset,
        window_size=mcfg["window_size"],
        username=args.username,
        password=args.password,
    )

    print(f"MQTT Gateway — dataset={args.dataset} broker={args.broker}:{args.port}")
    print(f"  Subscribe: iot/gateway/{args.dataset}/data")
    print(f"  Alerts:    iot/gateway/{args.dataset}/alerts")
    print(f"  Status:    iot/gateway/{args.dataset}/status")
    gateway.run()


if __name__ == "__main__":
    main()
