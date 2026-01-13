from __future__ import annotations

import pytest

from telephony.rtp import build_rtp_packet, parse_rtp_packet


def test_rtp_roundtrip_minimal_packet() -> None:
    payload = b"abc123"
    raw = build_rtp_packet(
        payload_type=0,
        sequence=42,
        timestamp=123456,
        ssrc=0x01020304,
        marker=True,
        payload=payload,
    )

    pkt = parse_rtp_packet(raw)
    assert pkt.payload_type == 0
    assert pkt.sequence == 42
    assert pkt.timestamp == 123456
    assert pkt.ssrc == 0x01020304
    assert pkt.marker is True
    assert pkt.payload == payload


@pytest.mark.parametrize("data", [b"", b"123", b"\x80\x00\x00\x01"]) 
def test_rtp_too_short_raises(data: bytes) -> None:
    with pytest.raises(ValueError):
        parse_rtp_packet(data)
