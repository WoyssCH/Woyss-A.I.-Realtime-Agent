# True Full-Duplex PSTN (Twilio -> SIP -> Asterisk -> Python)

This project already supports Twilio `<Gather>` (half-duplex) and an experimental Media Streams variant (still half-duplex for responses). For **true duplex** (AI can speak while listening / barge-in), the recommended path is:

PSTN → Twilio Voice webhook (TwiML) → **SIP** → **Asterisk** → **ExternalMedia (RTP/PCMU)** → Python media server (Whisper + local Coqui TTS)

## 1) Twilio setup

- Configure your Twilio Phone Number **Voice webhook** to call:
  - `POST https://<PUBLIC_BASE_URL>/api/twilio/sip`
- Set environment variable:
  - `TWILIO_SIP_TARGET_URI=sip:assistant@<your-sip-domain>.sip.twilio.com`

Notes:
- Your `<your-sip-domain>.sip.twilio.com` comes from Twilio SIP Domains / Trunking.
- Twilio must be able to reach your SIP endpoint (Asterisk). Typically you use Twilio SIP Domain termination pointed at your Asterisk public IP.

## 2) Asterisk (docker)

Asterisk runs in docker-compose as service `asterisk` with config templates mounted from `docker/asterisk`.

Required configs (minimal scaffold provided):
- `docker/asterisk/http.conf`: enables ARI + websocket
- `docker/asterisk/ari.conf`: creates ARI user
- `docker/asterisk/extensions.conf`: routes inbound calls into Stasis app `woyss`

You must:
- Set the password in `docker/asterisk/ari.conf` (default is `change_me`).
- Configure `docker/asterisk/pjsip.conf` for your Twilio SIP trunk/domain.

## 3) Python media + ARI controller

Docker services:
- `media`: runs `python -m telephony.media_server`
- `ari`: runs `python -m telephony.ari_controller`

Environment variables (example `.env`):

- `PUBLIC_BASE_URL=https://<your-public-url>`
- `TWILIO_SIP_TARGET_URI=sip:assistant@<your-sip-domain>.sip.twilio.com`

- `ASTERISK_ARI_URL=http://asterisk:8088/ari`
- `ASTERISK_ARI_USERNAME=woyss`
- `ASTERISK_ARI_PASSWORD=change_me`
- `ASTERISK_STASIS_APP=woyss`

- `MEDIA_SERVER_HOST=media`
- `MEDIA_SERVER_PORT=20000`

## 4) Run

- `docker compose up --build`
- Place a call to your Twilio number.

If the SIP side is wired correctly, you should see:
- `ari` logs: `StasisStart ... Bridged channel ... externalMedia ...`
- `media` logs: `New RTP stream ...`

## 5) Current limitations

This is a functional scaffold, not a production-grade PBX/media stack yet:
- No jitter buffer / packet loss concealment
- Simple chunking heuristic (no real VAD)
- No barge-in cancellation (AI speaking while user interrupts)

Next steps:
- Add VAD (or energy threshold) + partial ASR
- Add "cancel TTS on user speech" logic
- Add call correlation (CallSid ↔ channel ID ↔ RTP SSRC)
