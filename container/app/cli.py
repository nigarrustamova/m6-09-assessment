#!/usr/bin/env python3
# app/cli.py
# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint for the cat-detector container.
#
# Instructor usage (exact contract):
#   docker run --rm \
#     -v /path/to/holdout:/data/input:ro \
#     -v /path/to/results:/data/output \
#     rustamova-nigar-cat-detector:submission \
#     --input /data/input --output /data/output/predictions.json --threshold 0.25
#
# Info usage (optional helper):
#   docker run --rm rustamova-nigar-cat-detector:submission info
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import sys
from pathlib import Path

# ── fixed paths inside the container ─────────────────────────────────────────
STUDENT_JSON = Path("/app/STUDENT.json")
ONNX_MODEL   = Path("/app/models/best.onnx")

IMAGE_EXTS   = {".jpg", ".jpeg", ".png"}


def cmd_info():
    """Print STUDENT.json to stdout and exit 0."""
    print(STUDENT_JSON.read_text())


def cmd_predict(input_dir: Path, output_path: Path, threshold: float):
    """Run detection on every image in input_dir and write predictions JSON."""
    sys.path.insert(0, "/app")
    from app.detector import CatDetector

    # Load model once — expensive step, paid only once per run
    detector = CatDetector(
        onnx_path=str(ONNX_MODEL),
        imgsz=640,
        conf=threshold,          # honour the threshold passed by the instructor
        class_names=("cat",),
    )

    # Collect all image files recursively, sorted by filename (not full path)
    image_files = sorted(
        (p for p in input_dir.rglob("*")
         if p.suffix.lower() in IMAGE_EXTS and p.is_file()),
        key=lambda p: p.name,   # sort on filename only, as required
    )

    if not image_files:
        print("WARNING: no images found in", input_dir, file=sys.stderr)

    # Ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = []

    for img_file in image_files:
        try:
            boxes = detector.predict(str(img_file))
        except Exception as exc:
            print(f"ERROR on {img_file.name}: {exc}", file=sys.stderr)
            boxes = []

        # Build detections list — only include boxes at/above threshold
        detections = []
        for b in boxes:
            if b["confidence"] >= threshold:
                detections.append({
                    "bbox":  [b["xmin"], b["ymin"], b["xmax"], b["ymax"]],
                    "score": round(b["confidence"], 6),
                    "label": b["class"],
                })

        predictions.append({
            "image":      img_file.name,   # filename only, not full path
            "detections": detections,
        })

    output = {
        "model":       "yolo26-cat-onnx",
        "threshold":   threshold,
        "predictions": predictions,
    }

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    n_with = sum(1 for p in predictions if p["detections"])
    print(
        f"Wrote {len(predictions)} predictions to {output_path}  "
        f"({n_with} images with detections)"
    )


# ── dispatch ──────────────────────────────────────────────────────────────────

def main():
    # Allow `info` subcommand without any other flags
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "info":
        cmd_info()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="YOLO26 cat detector — batch inference"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing the batch of images (recursive scan)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Full path to the JSON file to write (e.g. /data/output/predictions.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Confidence floor; detections below this score are dropped (default: 0.25)",
    )

    args = parser.parse_args()
    cmd_predict(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()