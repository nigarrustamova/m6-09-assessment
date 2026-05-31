# app/detector.py
# ─────────────────────────────────────────────────────────────────────────────
# CatDetector: loads best.onnx once and exposes predict(image_path) -> list[dict]
# All coordinates returned are in original-image pixel space (top-left origin).
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    """
    ONNX-based cat detector wrapping a YOLO26 end-to-end export.

    The YOLO26 end-to-end head outputs a single tensor of shape (1, 300, 6):
        [x1, y1, x2, y2, confidence, class_id]
    with up to 300 detections per image, already NMS-filtered.
    Coordinates are in letterboxed-input-image pixel space and must be
    mapped back to original-image pixel space before returning.
    """

    def __init__(
        self,
        onnx_path: str,
        imgsz: int = 640,
        conf: float = 0.25,
        class_names: tuple = ("cat",),
    ):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.imgsz       = imgsz
        self.conf        = conf
        self.class_names = class_names
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # ── internal helpers ──────────────────────────────────────────────────────

    def _letterbox(self, img: Image.Image):
        """Resize img to (imgsz, imgsz) preserving aspect ratio with grey padding.

        Returns
        -------
        canvas   : PIL Image, RGB, (imgsz, imgsz)
        scale    : float — uniform scale factor applied to the original
        pad      : (pad_left, pad_top) — pixels of grey border added
        """
        orig_w, orig_h = img.size
        scale = min(self.imgsz / orig_w, self.imgsz / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas  = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        pad_left = (self.imgsz - new_w) // 2
        pad_top  = (self.imgsz - new_h) // 2
        canvas.paste(resized, (pad_left, pad_top))

        return canvas, scale, (pad_left, pad_top)

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> list:
        """Run detection on a single image.

        Parameters
        ----------
        image_path : str
            Absolute or relative path to the image file.

        Returns
        -------
        list of dict, each with keys:
            xmin, ymin, xmax, ymax  — float, original-image pixel coordinates
            confidence              — float in [0, 1]
            class                   — str, class name (e.g. "cat")
        Returns an empty list if no detections exceed the confidence threshold.
        """
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        canvas, scale, (pad_left, pad_top) = self._letterbox(img)

        # Build NCHW float32 tensor, normalised to [0, 1]
        x = (np.array(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        # Run ONNX session — output shape (1, 300, 6)
        raw = self.session.run([self.output_name], {self.input_name: x})[0][0]

        results = []
        for x1, y1, x2, y2, conf, cls in raw:
            if conf < self.conf:
                continue

            # Undo letterbox: remove padding offset, then undo scale
            x1 = (x1 - pad_left) / scale
            y1 = (y1 - pad_top)  / scale
            x2 = (x2 - pad_left) / scale
            y2 = (y2 - pad_top)  / scale

            # Clip to original image bounds
            x1 = float(max(0.0, min(orig_w, x1)))
            y1 = float(max(0.0, min(orig_h, y1)))
            x2 = float(max(0.0, min(orig_w, x2)))
            y2 = float(max(0.0, min(orig_h, y2)))

            results.append({
                "xmin":       x1,
                "ymin":       y1,
                "xmax":       x2,
                "ymax":       y2,
                "confidence": float(conf),
                "class":      self.class_names[int(cls)],
            })

        return results