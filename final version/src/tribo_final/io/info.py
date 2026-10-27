"""Scan metadata (.info, XML CRunDetail)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def read_info(path: Path | str) -> dict:
    out: dict = {}
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return out
    node = root.find("CRunDetail")
    if node is None:
        return out
    for child in node:
        text = (child.text or "").strip()
        try:
            out[child.tag] = float(text)
        except ValueError:
            out[child.tag] = text
    return out
