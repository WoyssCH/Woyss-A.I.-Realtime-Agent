from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
import websockets

from config.settings import get_settings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AriAuth:
    username: str
    password: str


class AriController:
    """Asterisk ARI controller to bridge SIP call audio to ExternalMedia.

    This is the control-plane component for the true-duplex architecture.

    It expects an Asterisk dialplan that sends channels into a Stasis app
    (see docs and the docker/asterisk templates).
    """

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.asterisk_ari_username or not settings.asterisk_ari_password:
            raise RuntimeError("ASTERISK_ARI_USERNAME/PASSWORD not configured")

        self._base_url = settings.asterisk_ari_url.rstrip("/")
        self._app = settings.asterisk_stasis_app
        self._external_host = f"{settings.media_server_host}:{settings.media_server_port}"
        self._external_format = "ulaw"
        self._auth = AriAuth(settings.asterisk_ari_username, settings.asterisk_ari_password)

    async def run_forever(self) -> None:
        """Connect to ARI events and manage bridges/external media channels."""

        ws_url = self._events_ws_url()
        LOGGER.info("Connecting to ARI events: %s", ws_url)

        async with httpx.AsyncClient(auth=(self._auth.username, self._auth.password), timeout=10.0) as client:
            async for ws in self._ws_connect_loop(ws_url):
                try:
                    await self._handle_events(ws, client)
                except Exception:
                    LOGGER.exception("ARI event loop crashed; reconnecting")

    async def _handle_events(self, ws: websockets.WebSocketClientProtocol, client: httpx.AsyncClient) -> None:
        async for message in ws:
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                continue

            event_type = str(event.get("type") or "")
            if event_type != "StasisStart":
                continue

            channel = event.get("channel") or {}
            channel_id = str(channel.get("id") or "")
            if not channel_id:
                continue

            LOGGER.info("StasisStart channel=%s", channel_id)
            await self._bridge_channel_to_external_media(client, channel_id)

    async def _bridge_channel_to_external_media(self, client: httpx.AsyncClient, channel_id: str) -> None:
        bridge_id = await self._create_bridge(client)
        external_channel_id = await self._create_external_media_channel(client)

        await self._add_channels_to_bridge(client, bridge_id, [channel_id, external_channel_id])
        LOGGER.info("Bridged channel=%s with externalMedia=%s (bridge=%s)", channel_id, external_channel_id, bridge_id)

    async def _create_bridge(self, client: httpx.AsyncClient) -> str:
        resp = await client.post(f"{self._base_url}/bridges", params={"type": "mixing"})
        resp.raise_for_status()
        data = resp.json()
        return str(data["id"])

    async def _create_external_media_channel(self, client: httpx.AsyncClient) -> str:
        params = {
            "app": self._app,
            "external_host": self._external_host,
            "format": self._external_format,
            "encapsulation": "rtp",
            "direction": "both",
        }
        resp = await client.post(f"{self._base_url}/channels/externalMedia", params=params)
        resp.raise_for_status()
        data = resp.json()
        return str(data["id"])

    async def _add_channels_to_bridge(self, client: httpx.AsyncClient, bridge_id: str, channel_ids: list[str]) -> None:
        params = {"channel": ",".join(channel_ids)}
        resp = await client.post(f"{self._base_url}/bridges/{bridge_id}/addChannel", params=params)
        resp.raise_for_status()

    def _events_ws_url(self) -> str:
        # ARI events WS endpoint: /ari/events?app=<app>&api_key=<user>:<pass>
        base = self._base_url
        if base.startswith("https://"):
            scheme = "wss://"
            rest = base.removeprefix("https://")
        elif base.startswith("http://"):
            scheme = "ws://"
            rest = base.removeprefix("http://")
        else:
            scheme = "ws://"
            rest = base

        query = urlencode({"app": self._app, "api_key": f"{self._auth.username}:{self._auth.password}"})
        return f"{scheme}{rest}/events?{query}"

    @staticmethod
    async def _ws_connect_loop(ws_url: str):
        while True:
            try:
                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                    yield ws
            except Exception:
                LOGGER.exception("Failed to connect to ARI websocket; retrying")
                await asyncio.sleep(2)


async def _amain() -> None:
    ctrl = AriController()
    await ctrl.run_forever()


def main() -> None:
    logging.basicConfig(level=get_settings().log_level)
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
