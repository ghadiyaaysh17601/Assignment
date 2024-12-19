# Video Shot Classification using EfficientFormerV2

## Overview
This Assignment implements a video shot classification pipeline using the EfficientFormerV2 architecture. It processes video files frame by frame, applies a trained classification model, and outputs timestamped results for detected shots in JSON format.

The system is optimized for multi-class classification and supports real-time processing on various video formats.

---

## Features

- **Video Shot Detection**: Classifies frames into predefined categories using a trained model.
- **Majority Voting**: Stabilizes predictions by applying majority voting over sliding windows.
- **JSON Output**: Saves results with timestamped start and end times for detected shots.
- **Error Handling**: Robust validation for model loading and video file processing.

---

## Requirements

- Python 3.9+
- Libraries:
  - `torch`
  - `timm`
  - `numpy`
  - `opencv-python`
  - `Pillow`

---

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure that the model checkpoint file is available in the specified path.

---

## Usage

To classify shots in a video file, run the following command:

```bash
python inference.py <video_path> --model_path <model_checkpoint_path>
```

### Example:
```bash
python inference.py sample_video.mp4 --model_path checkpoints/best_model.pth
```

### Parameters:
- `<video_path>`: Path to the video file.
- `<model_checkpoint_path>`: Path to the trained model checkpoint (default: `D:\Assignment\checkpoints\best_model.pth`).

### Output:
The results are saved in a JSON file named `<video_filename>_output.json` in the current directory.

---

## File Structure

```plaintext
video-shot-classification/
│
├── checkpoints/
│   └── best_model.pth        # Trained model checkpoint
├── inference.py              # Inference script
├── train.py                  # Training script
├── requirements.txt          # Required dependencies
├── README.md                 # Project documentation
└── sample_video.mp4          # Example video file (optional)
```

---

## Model Details

- **Architecture:** EfficientFormerV2 (`efficientformerv2_s0.snap_dist_in1k`).
- **Pretrained Weights:** Initialized with ImageNet weights.
- **Fine-Tuned On:** 7-class multi-class classification task.

---

## Output Format

Results are saved as a JSON file with the following structure:

```json
{
    "shots": [
       {
            "label": "Short_player",
            "start_time": 0.0
        },
        {
            "label": "long",
            "start_time": 1.98
        }
    ]
}
```

---


