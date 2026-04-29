#!/usr/bin/env python3
"""Live viewer for the BRX Isaac Lab HTTP camera endpoints."""

from __future__ import annotations

import argparse
import time
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show BRX head/left_wrist/right_wrist camera streams.")
    parser.add_argument("--brx_url", default="http://127.0.0.1:8765", help="BRX control server base URL.")
    parser.add_argument("--fps", type=float, default=10.0, help="Viewer refresh rate.")
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout in seconds.")
    parser.add_argument("--window_name", default="BRX cameras")
    parser.add_argument("--width", type=int, default=640, help="Displayed width per camera.")
    parser.add_argument("--height", type=int, default=480, help="Displayed height per camera.")
    return parser.parse_args()


def fetch_png(url: str, timeout: float) -> np.ndarray:
    with urlopen(url, timeout=timeout) as resp:
        data = np.frombuffer(resp.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode image from {url}")
    return image


def label_image(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (220, 34), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_error_panel(width: int, height: int, message: str) -> np.ndarray:
    panel = np.full((height, width, 3), 32, dtype=np.uint8)
    cv2.putText(panel, "camera unavailable", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, message[:70], (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    args = parse_args()
    base = args.brx_url.rstrip("/")
    cameras = [
        ("head", f"{base}/camera/head.png"),
        ("left_wrist", f"{base}/camera/left_wrist.png"),
        ("right_wrist", f"{base}/camera/right_wrist.png"),
    ]
    period = 1.0 / max(args.fps, 1e-6)
    print(f"[viewer] BRX URL: {base}")
    print("[viewer] press q or ESC to quit")

    while True:
        start = time.monotonic()
        panels = []
        for name, url in cameras:
            try:
                image = fetch_png(url, args.timeout)
                image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
                panels.append(label_image(image, name))
            except (URLError, TimeoutError, RuntimeError) as exc:
                panels.append(make_error_panel(args.width, args.height, str(exc)))

        canvas = np.concatenate(panels, axis=1)
        cv2.imshow(args.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        elapsed = time.monotonic() - start
        if elapsed < period:
            time.sleep(period - elapsed)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
