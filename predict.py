import os
import cv2
import torch
import numpy as np

from PIL import Image
import torchvision.transforms as T

from torchvision.models.video import (
    r2plus1d_18,
    R2Plus1D_18_Weights
)

# =========================================================
# CONFIG
# =========================================================

VIDEO_PATH = "../3dcnn/data/XD_violence_balanced/test/Shooting/Test_11_Shooting.mp4"

MODEL_PATH = "best_r2plus1d.pth"

NUM_FRAMES = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "Fighting",
    "Normal",
    "Shooting"
]

# =========================================================
# LOAD MODEL
# =========================================================

weights = R2Plus1D_18_Weights.DEFAULT

model = r2plus1d_18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(CLASSES)
)

# ---------------------------------------------------------
# LOAD TRAINED WEIGHTS
# ---------------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# support both:
# torch.save(model.state_dict())
# and checkpoint dict

if "model" in checkpoint:

    model.load_state_dict(
        checkpoint["model"]
    )

else:

    model.load_state_dict(checkpoint)

model = model.to(DEVICE)

model.eval()

print("Model loaded.")

# =========================================================
# NORMALIZATION
# =========================================================

MEAN = [0.43216, 0.394666, 0.37645]

STD = [0.22803, 0.22145, 0.216989]

# =========================================================
# LOAD VIDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

frames = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # BGR -> RGB
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame = Image.fromarray(frame)

    # -----------------------------------------------------
    # RESIZE
    # -----------------------------------------------------

    frame = T.functional.resize(
        frame,
        (112, 112)
    )

    # -----------------------------------------------------
    # TO TENSOR
    # -----------------------------------------------------

    frame = T.functional.to_tensor(frame)

    # -----------------------------------------------------
    # NORMALIZE
    # -----------------------------------------------------

    frame = T.functional.normalize(
        frame,
        mean=MEAN,
        std=STD
    )

    frames.append(frame)

cap.release()

# =========================================================
# CHECK FRAME COUNT
# =========================================================

if len(frames) < NUM_FRAMES:

    raise ValueError(
        f"Video too short. "
        f"Need at least {NUM_FRAMES} frames."
    )

print(f"Total frames: {len(frames)}")

# =========================================================
# SAMPLE CLIP
# =========================================================

# center clip

start_idx = (
    len(frames) - NUM_FRAMES
) // 2

clip = frames[
    start_idx : start_idx + NUM_FRAMES
]

# =========================================================
# STACK
# Shape:
# (T, C, H, W)
# =========================================================

video = torch.stack(clip)

# =========================================================
# MODEL EXPECTS:
# (C, T, H, W)
# =========================================================

video = video.permute(
    1,
    0,
    2,
    3
)

# =========================================================
# ADD BATCH DIMENSION
# Shape:
# (1, C, T, H, W)
# =========================================================

video = video.unsqueeze(0)

video = video.to(DEVICE)

# =========================================================
# PREDICTION
# =========================================================

with torch.no_grad():

    output = model(video)

    probs = torch.softmax(
        output,
        dim=1
    )

    pred_idx = torch.argmax(
        probs,
        dim=1
    ).item()

    confidence = probs[
        0,
        pred_idx
    ].item()

# =========================================================
# RESULTS
# =========================================================

print("\n================ RESULT ================\n")

print(f"Prediction : {CLASSES[pred_idx]}")

print(f"Confidence : {confidence:.4f}")

print("\nClass Probabilities:\n")

for i, cls in enumerate(CLASSES):

    print(
        f"{cls:<10}: "
        f"{probs[0,i].item():.4f}"
    )