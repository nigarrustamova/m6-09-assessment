# Cat Detection — YOLO26 Assessment

## Image for Leaderboard

```bash
docker pull nigarrustamova/cat-detector:final
```

Image: `nigarrustamova/cat-detector:final`  
Student: Nigar Rustamova

## Quick Test

```bash
# Confirm identity
docker run --rm nigarrustamova/cat-detector:final info

# Run predictions (replace paths with your own test folders)
docker run --rm \
  -v /absolute/path/to/images:/data/input:ro \
  -v /absolute/path/to/results:/data/output \
  nigarrustamova/cat-detector:final predict
```

## Model Details

| Field       | Value         |
|-------------|---------------|
| Framework   | YOLO26        |
| Variant     | yolo26s       |
| Input size  | 640 × 640     |
| Epochs      | 50            |
| mAP@0.5     | 0.9172        |
| Export      | ONNX opset 17 |
