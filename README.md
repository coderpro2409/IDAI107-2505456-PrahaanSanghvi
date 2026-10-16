# 🅿️ CurbSense Lite

Upload a parking-lot photo → a YOLOv8n model finds every bay, marks it **free** (blue) or **occupied** (orange), and the app reports occupancy %, a crowding rating and a short driver recommendation.

| Occupancy | Rating |
|---|---|
| under 40 % | Light |
| 40 - 75 % | Steady |
| over 75 % | Heavy |

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app (detection, rating, advice) |
| `prepare_data.py` | Turns the PKLot Roboflow COCO export into a YOLO dataset (`bay_data/`) |
| `train_model.ipynb` | Colab notebook that fine-tunes YOLOv8n and evaluates it |
| `bay_detector.pt` | Trained weights (you add this after training) |
| `requirements.txt`, `packages.txt` | Dependencies |

## Steps
1. Download PKLot (Roboflow 640 COCO export, [Kaggle](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset)) and put `train/`, `valid/`, `test/` beside these files.
2. `python prepare_data.py` → creates `bay_data/` and `bay_data.zip`.
3. Upload `bay_data.zip` to Colab, run `train_model.ipynb`, download the weights, save them here as `bay_detector.pt`.
4. `pip install -r requirements.txt` then `streamlit run app.py`.

## Results (fill in after training)
| Precision | Recall | mAP@50 | mAP@50-95 |
|---:|---:|---:|---:|
| [ ] | [ ] | [ ] | [ ] |

Dataset: PKLot, Almeida et al. (2015), Universidade Federal do Paraná, CC BY 4.0. Model: Ultralytics YOLOv8.
