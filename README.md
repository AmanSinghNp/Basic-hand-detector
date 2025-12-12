# Hand Tracker with AR Cube

I wanted to have some fun exploring the MediaPipe library and learning the basics of it, so I made this simple Python project for real-time hand tracking and basic Augmented Reality (AR) visualization.

## Overview

This application uses your webcam to detect hand gestures and overlay an interactive 3D cube. It demonstrates how to map physical hand movements to digital objects using computer vision.

## Features

- **Single Hand Tracking**: Focuses on the primary hand for stable control.
- **Pinch-to-Resize**: Change the cube's size by pinching your thumb and index finger.
- **3D Rotation**: Tilt and turn your hand to rotate the cube in 3D space.
- **Visual Effects**: Includes neon glow, particle systems, and motion trails.
- **Finger Controls**:
  - Middle Finger (Folded): Toggle solid faces.
  - Ring Finger (Folded): Cycle colors.
  - Pinky (Extended): Enable particles.

## Requirements

- Python 3.x
- Webcam

## Installation

1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   python hand_tracker.py

## Usage

- Select your camera index when prompted (usually 0).
- Show your hand to the camera.
- Pinch thumb and index finger to control the cube.
- Press 'q' to exit.

