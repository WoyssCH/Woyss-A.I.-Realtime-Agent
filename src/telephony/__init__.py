"""Telephony components for true full-duplex calls.

This package is intentionally separated from the Twilio Media Streams (WS) prototype.
The intended production architecture is:
PSTN -> Twilio -> SIP -> Asterisk -> ExternalMedia (RTP) -> Python media server.
"""
