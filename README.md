# AI Camera – TensorFlow Lite Vision Pipeline

## Overview
On-device computer vision pipeline that runs TensorFlow Lite inference on camera frames to classify
target position in real time.

## Approach
- Capture camera frames and preprocess to model input format
- Run inference using a TensorFlow Lite classification model
- Interpret outputs to determine target position/class
- Save frames and diagnostics to validate model performance

## Technologies
- Java
- TensorFlow Lite
- OpenCV

## Repo Notes
This repository contains the core inference and vision logic extracted from a larger Android
robotics codebase; platform-specific setup and hardware integration code is intentionally omitted
for clarity.
