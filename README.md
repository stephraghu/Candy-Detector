# Candy Detection with YOLO11

A Google Colab notebook workflow for downloading a custom candy dataset, splitting and configuring it for YOLO training, training a `yolo11s` object detection model, running validation predictions, packaging the trained artifacts, and streaming live webcam detections inside Colab.

## Overview

This notebook implements an end-to-end custom object detection pipeline using Ultralytics YOLO11:

- Downloads and extracts a candy dataset archive (`candy_data_06JAN25.zip`) to `/content/custom_data`.
- Splits the dataset into training and validation sets (`90%` train split via `train_val_split.py`) and generates `data.yaml`.
- Trains `yolo11s.pt` for `60` epochs at `640x640` resolution (`imgsz=640`).
- Runs detection inference on validation images (`data/validation/images`), previews up to 10 output `.jpg` predictions, exports `/content/my_model.zip`, and runs a browser-to-Colab webcam inference loop (`~15 fps`, confidence threshold `> 0.4`).

## Dataset and Configuration

- **Source archive**: Downloaded from `https://s3.us-west-1.amazonaws.com/evanjuras.com/resources/candy_data_06JAN25.zip` to `/content/data.zip` and extracted into `/content/custom_data`.
- **Train/validation split**: Prepared via `train_val_split.py` from `EdjeElectronics/Train-and-Deploy-YOLO-Models` with `--datapath /content/custom_data --train_pct=0.9`.
- **Dataset YAML (`data.yaml`)**: Generated from `/content/custom_data/classes.txt` with `path: /content/data`, `train: train/images`, `val: validation/images`, `nc` set to the non-empty class count, and `names` populated from `classes.txt`.

## Model Training, Export, and Live Webcam Inference

- **Training**: `yolo detect train data=/content/data.yaml model=yolo11s.pt epochs=60 imgsz=640`.
- **Validation prediction & preview**: Predicts on `data/validation/images` using `runs/detect/train/weights/best.pt` (`save=True`) and displays up to 10 `.jpg` files from `/content/runs/detect/predict/*.jpg` at `width=400`.
- **Packaging**: Copies `best.pt` to `/content/my_model/my_model.pt`, copies `/content/runs/detect/train` into `/content/my_model`, and archives them into `/content/my_model.zip`.
- **Webcam streaming**: Captures `640x480` frames via HTML5 `getUserMedia` and canvas JPEG encoding (`quality=0.8`), sends them to Python via `google.colab.kernel.invokeFunction('notebook.process_frame', [data], {})`, draws bounding boxes, confidence percentages, and an `Objects: <count>` overlay, and updates the live image display.

## Dependencies

- `ultralytics` (CLI `yolo` and Python `YOLO`)
- `PyYAML` (`yaml`)
- `opencv-python` (`cv2`)
- `numpy`
- `IPython` (`display`, `Javascript`, `Image`)
- `google.colab` (`eval_js`, `output`)
- Standard library modules (`os`, `glob`, `time`, `base64`)

## Usage

1. Verify GPU availability with `!nvidia-smi` (notebook metadata specifies a `T4` GPU accelerator).
2. Download and extract `candy_data_06JAN25.zip` into `/content/custom_data`, then run `train_val_split.py` with `--train_pct=0.9`.
3. Install `ultralytics` and run the `create_data_yaml` cell to write `/content/data.yaml` from `/content/custom_data/classes.txt`.
4. Train `yolo11s.pt` for `60` epochs, run `yolo detect predict` on `data/validation/images`, and inspect the displayed `.jpg` predictions.
5. Package `/content/my_model.zip`, optionally fetch `/content/yolo_detect.py`, and run the webcam streaming cell while granting browser camera permissions.

## Notes

- The unzip cell (`!unzip -q /content/data.zip -d /content/custom_data`) includes an interactive prompt output (`replace /content/custom_data/classes.txt? [y]es, [n]o, [A]ll, [N]one, [r]ename:`), so passing `-o` if re-running non-interactively avoids overwrite prompts.
- The `create_data_yaml` script writes `path: /content/data` in `/content/data.yaml` while the split command targets `--datapath /content/custom_data`; ensure the split output directory matches `/content/data` before training if needed.
- The webcam cell fetches `https://www.ejtech.io/code/yolo_detect.py` in the preceding cell without importing it directly inside the streaming script, which implements its own `YOLO` inference and `process_frame` callback.
