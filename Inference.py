import cv2
import torch
import timm
import json
import numpy as np
from torchvision import transforms
from PIL import Image
import argparse
from collections import Counter
import os
import sys


# Class names for labels
CLASS_NAMES = ["Closeup_head", "Closeup_waist", "long", "neg", "Short_player", "spectators_long", "spectators_short"]

def load_model(model_path):
    """Load the trained model with error handling."""
    try:
        model = timm.create_model('efficientformerv2_s0.snap_dist_in1k', pretrained=False, num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'. Please check the path.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

def validate_video_path(video_path):
    """Validate the video file path."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'. Please check the path.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(video_path):
        print(f"Error: '{video_path}' is not a file. Please provide a valid video file.", file=sys.stderr)
        sys.exit(1)
    if not video_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
        print(f"Error: Unsupported video format. Supported formats are .mp4, .avi, .mkv, .mov.", file=sys.stderr)
        sys.exit(1)

def predict_frame(frame, model, threshold=0.5):
    """Predict the class for a given frame with error handling."""
    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_transformed = transform(image_pil)  # Apply transformations
        image_tensor = image_transformed.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)

        # Apply threshold
        top_classes = [CLASS_NAMES[i] for i, prob in enumerate(probabilities) if prob > threshold]
        return top_classes if top_classes else [CLASS_NAMES[torch.argmax(probabilities).item()]]
    except Exception as e:
        print(f"Error during frame prediction: {e}", file=sys.stderr)
        return ["Unknown"]

def process_video(video_path, model, output_json_path, fps=25, threshold=0.5):
    """Process the video with error handling."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            raise ValueError("The video file contains no frames.")

        results = {"shots": []}
        prev_label = None
        frame_buffer = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Add the frame prediction to the buffer
            frame_buffer.append(predict_frame(frame, model, threshold))

            # Perform majority voting every 1 second
            if len(frame_buffer) >= actual_fps:
                # Flatten the list and get the most common class
                flat_predictions = [label for frame_preds in frame_buffer for label in frame_preds]
                most_common_label = Counter(flat_predictions).most_common(1)[0][0]

                # Get the start and end times in seconds
                end_time = frame_idx / actual_fps
                start_time = max(0.0, end_time - len(frame_buffer) / actual_fps)

                # If the label changes, add it to the results
                if most_common_label != prev_label:
                    results["shots"].append({
                        "label": most_common_label,
                        "start_time": round(start_time, 2)
                    })
                    prev_label = most_common_label

                frame_buffer.clear()  # Reset the buffer

            frame_idx += 1

        cap.release()

        # Save the results to a JSON file
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

    except ValueError as e:
        print(f"Video processing error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during video processing: {e}", file=sys.stderr)
        sys.exit(1)

# Main function with argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video for shot classification.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument(
        "--model_path", 
        default=r"D:\Assignment\checkpoints\best_model.pth",
        help="Path to the trained model checkpoint. Default: D:\\Assignment\\checkpoints\\best_model.pth"
    )
    args = parser.parse_args()

    # Validate video path
    validate_video_path(args.video_path)

    # Load the model
    model = load_model(args.model_path)

    # Prepare the transformation
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Automatically generate output JSON file name
    video_filename = os.path.splitext(os.path.basename(args.video_path))[0]
    output_json_path = 'D:\Assignment\Results\' + f"{video_filename}_output.json"

    # Process video
    process_video(
        video_path=args.video_path,
        model=model,
        output_json_path=output_json_path
    )
    print(f"Results saved to {output_json_path}")