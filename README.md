# Sperm Cell Detection with YOLOv8

Computer vision project for automated sperm cell detection and counting using YOLOv8.

## Project Overview

This project uses YOLOv8 to detect and count sperm cells in microscopic images. It can process individual images, batches of images, and videos.

## What I Built

- Trained YOLOv8 models (nano and medium) on sperm cell dataset
- Image detection with counting and density calculation
- Video processing for frame-by-frame analysis
- Batch processing for multiple images

## Dataset

- Training images: 240
- Validation images: 24
- Test images: 11
- Source: Roboflow Sperm Morphology Dataset

## Results

### YOLOv8n (Nano Model)
- mAP@50: 71.7%
- Precision: 67.3%
- Recall: 68.0%

### YOLOv8m (Medium Model)
- Better accuracy with more parameters
- Training visualizations available in `Sperm_counting/yolov8m_improved/`

## Project Structure

```
sperm-classification/
├── Full-Sperm-Cell-Detection-6/    # Dataset
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── Sperm_counting/                  # Training results
│   ├── yolov8n_training/
│   ├── yolov8m_improved/
│   └── yolov8m_improved2/
├── sprem_classification.ipynb       # Main notebook
└── output_video.mp4                 # Sample output
```

## How to Use

### 1. Install Requirements
```bash
pip install ultralytics opencv-python numpy pandas matplotlib roboflow jupyter
```

### 2. Train Model
```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
model.train(
    data='./Full-Sperm-Cell-Detection-6/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)
```

### 3. Test on Images
```python
model = YOLO('./Sperm_counting/yolov8m_improved/weights/best.pt')
results = model.predict('test_image.jpg', save=True)
print(f"Detected: {len(results[0].boxes)} sperm cells")
```

### 4. Process Video
```python
results = model.predict('video.mp4', save=True)
```

## Features

- ✅ Sperm cell detection
- ✅ Automatic counting
- ✅ Density calculation
- ✅ Batch processing
- ✅ Video analysis
- ✅ Training visualizations

## Technologies Used

- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy, Pandas
- Matplotlib
- Jupyter Notebook

## Future Improvements

- Morphology classification
- Motility tracking
- Web interface
- Mobile app

## License

MIT License

## Author

Elsaraf
- GitHub: @Elsaraf1
