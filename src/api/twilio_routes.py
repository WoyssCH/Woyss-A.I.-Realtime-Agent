"""Twilio Voice integration.

This module provides:
- Voice webhook (TwiML) for inbound/outbound calls.
- Recall endpoint to initiate outbound calls.

The voice flow is speech-to-text via Twilio <Gather input="speech">.
"""

from __future__ import annotations

import logging
from typing import Annotated
from urllib.parse import urlencode
from xml.sax.saxutils import escape

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from agents.errors import AssistantError
from api.dependencies import get_agent
from config.settings import get_settings
from integrations.twilio_client import build_twilio_client, get_twilio_config
from integrations.twilio_streaming import GLOBAL_TWILIO_STREAM_STORE, parse_twilio_ws_message

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/twilio", tags=["twilio"])


def _twiml_dial_sip(*, sip_uri: str) -> str:
    sip = escape(sip_uri)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Response>"
        "<Dial>"
        f"<Sip>{sip}</Sip>"
        "</Dial>"
        "</Response>"
    )


def _voice_url(request: Request) -> str:
    settings = get_settings()
    if settings.public_base_url:
        return f"{settings.public_base_url.rstrip('/')}/api/twilio/voice"
    return str(request.url_for("twilio_voice_webhook"))


def _twiml_response(xml: str) -> Response:
    # Twilio expects application/xml
    return Response(content=xml, media_type="application/xml")


def _twiml_gather(*, say_text: str, action_url: str, language: str) -> str:
    say = escape(say_text)
    action = escape(action_url)
    lang = escape(language)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Response>"
        f"<Gather input=\"speech\" action=\"{action}\" method=\"POST\" language=\"{lang}\" speechTimeout=\"auto\">"
        f"<Say language=\"{lang}\">{say}</Say>"
        "</Gather>"
        f"<Say language=\"{lang}\">Ich habe nichts gehoert.</Say>"
        "</Response>"
    )


def _twiml_say_and_loop(*, say_text: str, loop_url: str, language: str) -> str:
    say = escape(say_text)
    redirect = escape(loop_url)
    lang = escape(language)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Response>"
        f"<Say language=\"{lang}\">{say}</Say>"
        f"<Redirect method=\"POST\">{redirect}</Redirect>"
        "</Response>"
    )


def _to_ws_url(http_url: str) -> str:
    if http_url.startswith("https://"):
        return "wss://" + http_url.removeprefix("https://")
    if http_url.startswith("http://"):
        return "ws://" + http_url.removeprefix("http://")
    return http_url


def _twiml_stream(*, stream_url: str, pause_seconds: int, result_url: str, language: str) -> str:
    stream = escape(stream_url)
    result = escape(result_url)
    pause = max(1, int(pause_seconds))
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Response>"
        "<Connect>"
        f"<Stream url=\"{stream}\" />"
        "</Connect>"
        f"<Pause length=\"{pause}\" />"
        f"<Redirect method=\"POST\">{result}</Redirect>"
        "</Response>"
    )


@router.post("/voice")
async def twilio_voice_webhook(
    request: Request,
    agent=Depends(get_agent),
) -> Response:
    settings = get_settings()
    if not settings.twilio_enable_gather_voice:
        raise HTTPException(status_code=404, detail="Legacy voice webhook disabled")
    form = await request.form()

    call_sid = str(form.get("CallSid") or "").strip() or "unknown"
    speech_result = str(form.get("SpeechResult") or "").strip()

    language = settings.twilio_say_language
    url = _voice_url(request)

    # First webhook hit: ask for speech.
    if not speech_result:
        prompt = "Guten Tag. Wie kann ich Ihnen helfen?"
        return _twiml_response(_twiml_gather(say_text=prompt, action_url=url, language=language))

    try:
        result = await agent.handle_text(
            call_sid,
            speaker="patient",
            text=speech_result,
            language="de",
            attributes={"twilio": {"call_sid": call_sid}},
        )
    except AssistantError as exc:
        LOGGER.exception("Twilio voice handling failed: %s", exc)
        return _twiml_response(
            _twiml_say_and_loop(
                say_text="Entschuldigung, es gab ein Problem. Bitte versuchen Sie es erneut.",
                loop_url=url,
                language=language,
            )
        )

    return _twiml_response(
        _twiml_say_and_loop(
            say_text=result["assistant_response"],
            loop_url=url,
            language=language,
        )
    )


@router.post("/sip")
async def twilio_sip_dial() -> Response:
    """TwiML webhook that connects the call to a SIP target.

    This is the entry point for the true full-duplex path:
    PSTN -> Twilio -> SIP -> (Asterisk) -> ExternalMedia -> Python media server.
    """

    settings = get_settings()
    if not settings.twilio_sip_target_uri:
        raise HTTPException(status_code=404, detail="SIP dialing not configured")

    return _twiml_response(_twiml_dial_sip(sip_uri=settings.twilio_sip_target_uri))


@router.post("/voice_stream")
async def twilio_voice_stream_start(request: Request) -> Response:
    settings = get_settings()
    if not settings.twilio_enable_media_streams:
        raise HTTPException(status_code=404, detail="Media Streams disabled")

    form = await request.form()
    call_sid = str(form.get("CallSid") or "").strip() or "unknown"

    # WebSocket endpoint must be publicly reachable (wss:// recommended).
    if settings.public_base_url:
        base = settings.public_base_url.rstrip("/")
        stream_url = _to_ws_url(f"{base}/api/twilio/stream") + "?" + urlencode({"callSid": call_sid})
        result_url = f"{base}/api/twilio/voice_stream_result"
    else:
        # Fallback to request host. This may not work behind proxies; prefer PUBLIC_BASE_URL.
        stream_url = (
            _to_ws_url(str(request.base_url).rstrip("/") + "/api/twilio/stream")
            + "?"
            + urlencode({"callSid": call_sid})
        )
        result_url = str(request.url_for("twilio_voice_stream_result"))

    return _twiml_response(
        _twiml_stream(
            stream_url=stream_url,
            pause_seconds=settings.twilio_stream_pause_seconds,
            result_url=result_url,
            language=settings.twilio_say_language,
        )
    )


@router.post("/voice_stream_result")
async def twilio_voice_stream_result(
    request: Request,
    agent=Depends(get_agent),
) -> Response:
    settings = get_settings()
    if not settings.twilio_enable_media_streams:
        raise HTTPException(status_code=404, detail="Media Streams disabled")

    form = await request.form()
    call_sid = str(form.get("CallSid") or "").strip() or "unknown"

    wav_bytes = await GLOBAL_TWILIO_STREAM_STORE.pop_audio_wav(call_sid)
    if not wav_bytes:
        # No audio captured yet; keep streaming.
        loop_url = str(request.url_for("twilio_voice_stream_start"))
        return _twiml_response(
            _twiml_say_and_loop(
                say_text="Ich habe noch nichts gehoert. Bitte sprechen Sie nochmals.",
                loop_url=loop_url,
                language=settings.twilio_say_language,
            )
        )

    try:
        transcript, lang = await agent.transcribe_audio_bytes(wav_bytes, language_hint=None)
        result = await agent.handle_text(
            call_sid,
            speaker="patient",
            text=transcript,
            language=(lang or "de"),
            attributes={"twilio": {"call_sid": call_sid, "streaming": True}},
        )
    except AssistantError as exc:
        LOGGER.exception("Twilio streaming processing failed: %s", exc)
        result = {"assistant_response": "Entschuldigung, das hat nicht geklappt."}

    loop_url = str(request.url_for("twilio_voice_stream_start"))
    return _twiml_response(
        _twiml_say_and_loop(
            say_text=result["assistant_response"],
            loop_url=loop_url,
            language=settings.twilio_say_language,
        )
    )


@router.websocket("/stream")
async def twilio_media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    call_sid = websocket.query_params.get("callSid") or "unknown"
    try:
        while True:
            message = await websocket.receive_text()
            parsed = parse_twilio_ws_message(message)
            await GLOBAL_TWILIO_STREAM_STORE.handle_event(call_sid, parsed)
    except WebSocketDisconnect:
        return


class RecallRequest(BaseModel):
    to_number: str = Field(description="E.164 phone number, e.g. +4179...")


class RecallResponse(BaseModel):
    call_sid: str
    to_number: str


def get_twilio_client():
    return build_twilio_client()


def get_twilio_cfg():
    return get_twilio_config()


@router.post("/recalls", response_model=RecallResponse)
async def create_recall(
    payload: RecallRequest,
    x_api_key: Annotated[str | None, Header()] = None,
    twilio_client=Depends(get_twilio_client),
    cfg=Depends(get_twilio_cfg),
) -> RecallResponse:
    settings = get_settings()

    if settings.twilio_recall_api_key and x_api_key != settings.twilio_recall_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    voice_url = f"{cfg.public_base_url.rstrip('/')}/api/twilio/voice"

    call = twilio_client.calls.create(
        to=payload.to_number,
        from_=cfg.from_number,
        url=voice_url,
        method="POST",
    )

    return RecallResponse(call_sid=str(call.sid), to_number=payload.to_number)
