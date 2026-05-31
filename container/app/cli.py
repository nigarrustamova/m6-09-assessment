#!/usr/bin/env python3
# app/cli.py
# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint for the cat-detector container.
#
# Usage:
#   python /app/app/cli.py info
#   python /app/app/cli.py predict
#
# The ENTRYPOINT in the Dockerfile points here, so the subcommand is passed
# as the first positional argument:
#   docker run --rm <image> info
#   docker run --rm -v ...:/data/input:ro -v ...:/data/output <image> predict
# ─────────────────────────────────────────────────────────────────────────────

import csv
import json
import sys
from pathlib import Path

# ── paths fixed inside the container ─────────────────────────────────────────
STUDENT_JSON = Path("/app/STUDENT.json")
ONNX_MODEL   = Path("/app/models/best.onnx")
INPUT_DIR    = Path("/data/input")
OUTPUT_DIR   = Path("/data/output")
OUTPUT_CSV   = OUTPUT_DIR / "predictions.csv"

IMAGE_EXTS   = {".jpg", ".jpeg", ".png"}


def cmd_info():
    """Print STUDENT.json to stdout and exit 0."""
    print(STUDENT_JSON.read_text())


def cmd_predict():
    """Run detection on every image in /data/input/ and write predictions.csv."""
    # Import here so the info subcommand has zero heavy dependencies
    import sys
    sys.path.insert(0, "/app")
    from app.detector import CatDetector

    # Load model once — expensive step (~0.5 s), paid only once per run
    detector = CatDetector(
        onnx_path=str(ONNX_MODEL),
        imgsz=640,
        conf=0.25,
        class_names=("cat",),
    )

    # Collect all image files recursively
    image_files = sorted(
        p for p in INPUT_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )

    if not image_files:
        print("WARNING: no images found in /data/input/", file=sys.stderr)

    # Ensure output directory exists (it is mounted, but be safe)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CSV_HEADER = ["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()

        for img_file in image_files:
            # Build relative path with forward slashes (as required by the schema)
            rel_path = img_file.relative_to(INPUT_DIR).as_posix()

            try:
                boxes = detector.predict(str(img_file))
            except Exception as exc:
                # Log the error but do not abort — write an empty row for this image
                print(f"ERROR on {rel_path}: {exc}", file=sys.stderr)
                boxes = []

            if boxes:
                for b in boxes:
                    writer.writerow({
                        "image_path": rel_path,
                        "xmin":       f"{b['xmin']:g}",
                        "ymin":       f"{b['ymin']:g}",
                        "xmax":       f"{b['xmax']:g}",
                        "ymax":       f"{b['ymax']:g}",
                        "confidence": f"{b['confidence']:.6f}",
                        "class":      b["class"],
                    })
            else:
                # No detections — write a single row with empty bbox fields
                writer.writerow({
                    "image_path": rel_path,
                    "xmin": "", "ymin": "", "xmax": "", "ymax": "",
                    "confidence": "", "class": "",
                })

    print(f"Predictions written to {OUTPUT_CSV}  ({len(image_files)} images processed)")


# ── dispatch ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: cli.py <info|predict>", file=sys.stderr)
        sys.exit(1)

    subcmd = sys.argv[1].lower()

    if subcmd == "info":
        cmd_info()
    elif subcmd == "predict":
        cmd_predict()
    else:
        print(f"Unknown subcommand: {subcmd!r}. Use 'info' or 'predict'.", file=sys.stderr)
        sys.exit(1)