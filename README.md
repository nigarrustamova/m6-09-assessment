# Cat Detection — YOLO26 Assessment

**Student:** Nigar Rustamova  
**Docker Hub Image (Leaderboard):** `nigarrustamova/cat-detector:final`  
**Local Submission Tag:** `rustamova-nigar-cat-detector:submission`

---

## Docker Pull Command (for Leaderboard)

```bash
docker pull nigarrustamova/cat-detector:final
```

## Run (instructor contract)

```bash
docker run --rm \
  -v /path/to/holdout:/data/input:ro \
  -v /path/to/results:/data/output \
  rustamova-nigar-cat-detector:submission \
  --input /data/input \
  --output /data/output/predictions.json \
  --threshold 0.25
```

## Run (info — print student record)

```bash
docker run --rm rustamova-nigar-cat-detector:submission info
```

---

## Model Details

| Field       | Value              |
|-------------|--------------------|
| Framework   | YOLO26             |
| Variant     | yolo26s            |
| Input size  | 640 × 640          |
| Epochs      | 50                 |
| mAP@0.5     | 0.9172             |
| Export      | ONNX opset 17      |
