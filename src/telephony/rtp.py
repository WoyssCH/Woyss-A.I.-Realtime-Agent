from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RtpPacket:
    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int
    marker: bool
    payload: bytes


def parse_rtp_packet(data: bytes) -> RtpPacket:
    """Parse a minimal RTP packet (no CSRC, no header extensions).

    Raises:
        ValueError: if packet is too short or uses unsupported header features.
    """

    if len(data) < 12:
        raise ValueError("RTP packet too short")

    b0 = data[0]
    version = b0 >> 6
    padding = (b0 >> 5) & 1
    extension = (b0 >> 4) & 1
    csrc_count = b0 & 0x0F

    if version != 2:
        raise ValueError(f"Unsupported RTP version: {version}")
    if padding or extension or csrc_count:
        raise ValueError("RTP features not supported (padding/extension/CSRC)")

    b1 = data[1]
    marker = bool((b1 >> 7) & 1)
    payload_type = b1 & 0x7F

    sequence = int.from_bytes(data[2:4], "big")
    timestamp = int.from_bytes(data[4:8], "big")
    ssrc = int.from_bytes(data[8:12], "big")
    payload = data[12:]

    return RtpPacket(
        payload_type=payload_type,
        sequence=sequence,
        timestamp=timestamp,
        ssrc=ssrc,
        marker=marker,
        payload=payload,
    )


def build_rtp_packet(
    *,
    payload_type: int,
    sequence: int,
    timestamp: int,
    ssrc: int,
    marker: bool,
    payload: bytes,
) -> bytes:
    """Build a minimal RTP packet (no CSRC, no extensions)."""

    b0 = (2 << 6)  # V=2, P=0, X=0, CC=0
    b1 = ((1 if marker else 0) << 7) | (payload_type & 0x7F)

    header = bytes([b0, b1])
    header += int(sequence & 0xFFFF).to_bytes(2, "big")
    header += int(timestamp & 0xFFFFFFFF).to_bytes(4, "big")
    header += int(ssrc & 0xFFFFFFFF).to_bytes(4, "big")

    return header + payload
