#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local PaddleOCR worker for factory sign OCR.

Protocol: one JSON object per stdin line, one JSON object per stdout line.
Images are JPEG bytes encoded as base64. No network API is used.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time


class LocalPaddleOCR:
    def __init__(self, lang: str, model_name: str, min_score: float) -> None:
        self.lang = lang
        self.model_name = model_name
        self.min_score = float(min_score)
        self.engine = self._create_engine()

    def recognize(self, image_b64: str):
        import cv2
        import numpy as np

        data = base64.b64decode(image_b64.encode("ascii"))
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("cv2.imdecode failed")
        return self._recognize_image(image)

    def _create_engine(self):
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            raise RuntimeError("Cannot import paddleocr.PaddleOCR: {}".format(exc))

        attempts = [
            {
                "lang": self.lang,
                "ocr_version": self.model_name,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            },
            {
                "lang": self.lang,
                "ocr_version": self.model_name,
                "use_angle_cls": False,
                "show_log": False,
            },
            {
                "lang": self.lang,
                "use_angle_cls": False,
                "show_log": False,
            },
            {"lang": self.lang},
        ]
        last_error = None
        for kwargs in attempts:
            try:
                return PaddleOCR(**kwargs)
            except Exception as exc:
                last_error = exc
        raise RuntimeError("Failed to initialize PaddleOCR locally: {}".format(last_error))

    def _recognize_image(self, image):
        if hasattr(self.engine, "predict"):
            try:
                parsed = self._parse_predict_result(self.engine.predict(input=image))
                if parsed:
                    return parsed
            except TypeError:
                try:
                    parsed = self._parse_predict_result(self.engine.predict(image))
                    if parsed:
                        return parsed
                except Exception:
                    pass
            except Exception:
                pass
        return self._parse_legacy_result(self.engine.ocr(image, cls=False))

    def _parse_predict_result(self, result):
        texts = []
        for page in result or []:
            data = self._extract_page_data(page)
            rec_texts = self._as_list(data.get("rec_texts", data.get("text", [])))
            rec_scores = self._as_list(data.get("rec_scores", data.get("scores", [])))
            rec_boxes = self._as_list(data.get("rec_boxes", data.get("dt_polys", data.get("boxes", []))))
            for idx, text in enumerate(rec_texts or []):
                score = self._safe_float(rec_scores[idx] if idx < len(rec_scores) else 1.0)
                if score < self.min_score:
                    continue
                texts.append({
                    "text": str(text),
                    "score": score,
                    "box": self._jsonable(rec_boxes[idx]) if idx < len(rec_boxes) else [],
                })
        return texts

    def _extract_page_data(self, page):
        if isinstance(page, dict):
            return page
        data = getattr(page, "json", None)
        if isinstance(data, dict):
            return data
        if callable(data):
            value = data()
            if isinstance(value, dict):
                return value
        for attr in ("res", "result"):
            value = getattr(page, attr, None)
            if isinstance(value, dict):
                return value
        return {}

    def _parse_legacy_result(self, result):
        texts = []
        pages = result or []
        for page in pages:
            if not page:
                continue
            for item in page:
                if not item or len(item) < 2:
                    continue
                box = item[0]
                text_score = item[1]
                if isinstance(text_score, (list, tuple)) and text_score:
                    text = str(text_score[0])
                    score = self._safe_float(text_score[1] if len(text_score) > 1 else 1.0)
                else:
                    text = str(text_score)
                    score = 1.0
                if score < self.min_score:
                    continue
                texts.append({"text": text, "score": score, "box": self._jsonable(box)})
        return texts

    @staticmethod
    def _safe_float(value) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _jsonable(value):
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass
        if isinstance(value, (list, tuple)):
            return [LocalPaddleOCR._jsonable(item) for item in value]
        try:
            return float(value)
        except Exception:
            return value


def emit(payload) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--model-name", default="PP-OCRv6")
    parser.add_argument("--min-score", type=float, default=0.45)
    args = parser.parse_args()

    try:
        ocr = LocalPaddleOCR(args.lang, args.model_name, args.min_score)
        emit({"type": "ready", "ok": True, "model_name": args.model_name, "lang": args.lang})
    except Exception as exc:
        emit({"type": "ready", "ok": False, "error": str(exc)})
        return 2

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        started = time.time()
        try:
            req = json.loads(line)
            if req.get("cmd") == "shutdown":
                emit({"id": req.get("id"), "ok": True, "type": "shutdown"})
                return 0
            texts = ocr.recognize(req["image"])
            raw_text = " ".join(item.get("text", "") for item in texts if item.get("text"))
            emit({
                "id": req.get("id"),
                "ok": True,
                "texts": texts,
                "raw_text": raw_text,
                "elapsed_ms": int((time.time() - started) * 1000),
            })
        except Exception as exc:
            emit({
                "id": None,
                "ok": False,
                "texts": [],
                "raw_text": "",
                "elapsed_ms": int((time.time() - started) * 1000),
                "error": str(exc),
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
