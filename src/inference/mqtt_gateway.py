"""MQTT-based IoT Gateway for real-time anomaly detection.

Protocol flow:
    IoT Device  →  MQTT Broker  →  Gateway (this module)
    Gateway subscribes to a data topic, receives JSON sensor readings,
    runs the detector, and publishes results to an alert topic.

Topics (configurable):
    Subscribe: iot/gateway/{dataset}/data     — incoming raw samples
    Publish:   iot/gateway/{dataset}/alerts   — detection results
    Publish:   iot/gateway/{dataset}/status   — periodic stats

Payload format (inbound):
    {"features": [f1, f2, ..., fn], "device_id": "sensor_01"}

Payload format (outbound alert):
    {"label": "ATTACK", "error": 0.045, "threshold": 0.003,
     "device_id": "sensor_01", "window_id": 42, "timestamp": "..."}
"""
from __future__ import annotations
import json
import os
import sys
import time
import logging
from collections import deque
from datetime import datetime, UTC
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MQTTGateway:
    """Subscribes to MQTT data topic, detects anomalies, publishes alerts."""

    def __init__(
        self,
        detector,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        dataset: str = "ciciot",
        window_size: int = 64,
        username: Optional[str] = None,
        password: Optional[str] = None,
        keepalive: int = 60,
    ):
        self.detector    = detector
        self.dataset     = dataset
        self.window_size = window_size
        self.broker_host = broker_host
        self.broker_port = broker_port

        self.data_topic   = f"iot/gateway/{dataset}/data"
        self.alert_topic  = f"iot/gateway/{dataset}/alerts"
        self.status_topic = f"iot/gateway/{dataset}/status"

        self._buf        = deque(maxlen=window_size)
        self._window_id  = 0
        self._n_attack   = 0
        self._n_normal   = 0
        self._start_time = None

        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise ImportError("paho-mqtt not installed. Run: pip install paho-mqtt")

        self._client = mqtt.Client(client_id=f"iot-gateway-{dataset}")
        if username:
            self._client.username_pw_set(username, password)

        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

        self._broker_host = broker_host
        self._broker_port = broker_port
        self._keepalive   = keepalive

    # ── MQTT callbacks ────────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Connected to MQTT broker {self._broker_host}:{self._broker_port}")
            client.subscribe(self.data_topic)
            logger.info(f"Subscribed to: {self.data_topic}")
        else:
            logger.error(f"MQTT connection failed, rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        logger.warning(f"Disconnected from broker (rc={rc}). Will auto-reconnect.")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            features  = payload.get("features", [])
            device_id = payload.get("device_id", "unknown")

            sample = np.array(features, dtype=np.float32)
            self._buf.append(sample)

            if len(self._buf) < self.window_size:
                return  # not enough data yet

            window = np.array(self._buf, dtype=np.float32)
            result = self.detector.detect(window)
            result["window_id"] = self._window_id
            result["device_id"] = device_id
            result["timestamp"] = datetime.now(UTC).isoformat()

            self._window_id += 1
            if result["label"] == "ATTACK":
                self._n_attack += 1
            else:
                self._n_normal += 1

            self._client.publish(
                self.alert_topic,
                json.dumps(result),
                qos=1,
            )

            if result["label"] == "ATTACK":
                logger.warning(
                    f"ATTACK detected | device={device_id} "
                    f"error={result['reconstruction_error']:.4f} > "
                    f"threshold={result['threshold']:.4f}"
                )

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, status_interval: int = 30):
        """Connect and block forever, publishing status every status_interval seconds."""
        self._start_time = time.time()
        self._client.connect(self._broker_host, self._broker_port, self._keepalive)
        self._client.loop_start()

        logger.info(f"Gateway running — listening on '{self.data_topic}'")
        try:
            while True:
                time.sleep(status_interval)
                self._publish_status()
        except KeyboardInterrupt:
            logger.info("Shutting down MQTT gateway.")
        finally:
            self._client.loop_stop()
            self._client.disconnect()

    def _publish_status(self):
        elapsed = time.time() - (self._start_time or time.time())
        status = {
            "total_windows":    self._window_id,
            "n_normal":         self._n_normal,
            "n_attack":         self._n_attack,
            "elapsed_seconds":  round(elapsed, 1),
            "windows_per_second": round(self._window_id / elapsed, 1) if elapsed > 0 else 0,
            "timestamp":        datetime.now(UTC).isoformat(),
        }
        self._client.publish(self.status_topic, json.dumps(status), qos=0)
        logger.info(f"Status: {status}")
