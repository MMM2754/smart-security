# ─────────────────────────────────────────────
#  alerts/mqtt_publisher.py
#  Publishes classified events to MQTT broker
# ─────────────────────────────────────────────

import json
import time
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MQTT_BROKER, MQTT_PORT, MQTT_CLIENT_ID,
    MQTT_TOPIC_ALERTS, MQTT_TOPIC_WARNINGS, MQTT_TOPIC_STATUS,
    AlertLevel
)

try:
    import paho.mqtt.client as mqtt
    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed. Alerts will only be logged.")


class MQTTPublisher:
    def __init__(self):
        self._client    = None
        self._connected = False

        if not _MQTT_AVAILABLE:
            return

        try:
            self._client = mqtt.Client(client_id=MQTT_CLIENT_ID)
            self._client.on_connect    = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            self._client.loop_start()
            time.sleep(0.5)   # give it a moment to connect
        except Exception as e:
            logger.warning(f"MQTT broker not available at {MQTT_BROKER}:{MQTT_PORT}: {e}")
            logger.info("Continuing without MQTT — events are still stored in DB.")
            self._client = None

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.success(f"MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
            client.publish(MQTT_TOPIC_STATUS,
                           json.dumps({"status": "online", "ts": time.time()}))
        else:
            logger.error(f"MQTT connection failed: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        logger.warning("MQTT disconnected")

    def publish_event(self, event_dict: dict, verdict: dict, description: str):
        """Publish a classified event to the appropriate MQTT topic."""
        level = verdict.get("final_alert_level", event_dict.get("alert_level", "ORANGE"))

        payload = {
            "event_id":         event_dict.get("event_id"),
            "timestamp":        time.time(),
            "alert_level":      level,
            "event_type":       event_dict.get("event_type"),
            "description":      description,
            "zone_id":          event_dict.get("zone_id"),
            "track_id":         event_dict.get("track_id"),
            "face_hash":        event_dict.get("face_hash"),
            "recommended_action": verdict.get("recommended_action"),
            "confidence":       verdict.get("confidence"),
            "source_video":     event_dict.get("source_video"),
        }

        topic = self._topic_for_level(level)
        msg   = json.dumps(payload)

        # Always log to console regardless of MQTT
        level_icons = {
            AlertLevel.GREEN:  "🟢",
            AlertLevel.YELLOW: "🟡",
            AlertLevel.ORANGE: "🟠",
            AlertLevel.RED:    "🔴",
        }
        icon = level_icons.get(level, "⚪")
        logger.info(f"{icon} [{level}] {description}")

        if self._client and self._connected:
            try:
                result = self._client.publish(topic, msg, qos=1)
                if result.rc != 0:
                    logger.warning(f"MQTT publish failed: rc={result.rc}")
            except Exception as e:
                logger.warning(f"MQTT publish error: {e}")

        return payload

    def _topic_for_level(self, level: str) -> str:
        if level == AlertLevel.YELLOW:
            return MQTT_TOPIC_WARNINGS
        elif level in (AlertLevel.ORANGE, AlertLevel.RED):
            return MQTT_TOPIC_ALERTS
        return MQTT_TOPIC_STATUS

    def publish_status(self, status: dict):
        if self._client and self._connected:
            self._client.publish(MQTT_TOPIC_STATUS, json.dumps(status), qos=0)

    def disconnect(self):
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            logger.info("MQTT disconnected cleanly.")
