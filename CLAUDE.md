# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LectinAI is a Streamlit-based web application for AI-powered morphometric analysis of lectin staining in biological tissue samples. It combines classical image processing (color deconvolution for H&E DAB staining separation) with deep learning (ResNet18-based multi-task classification) to quantify staining intensity and positivity percentage.

## Tech Stack

- **UI Framework**: Streamlit (v1.30.0+) — Python web UI
- **Deep Learning**: PyTorch + torchvision (ResNet18 backbone)
- **Image Processing**: OpenCV, NumPy, PIL
- **Package Manager**: uv

## Common Commands

```bash
# Run the app
uv run streamlit run src/main.py   # Web browser (localhost:8501)
```

## Architecture

All source code lives in `src/`. There are no tests or CI/CD.

### Processing Pipeline

1. **Input**: Histological image (H&E DAB stained tissue)
2. Color deconvolution → separate Hematoxylin/DAB channels
3. Tissue mask via grayscale thresholding
4. Positive (DAB) areas via optical density threshold (0.25)
5. Border/inner zones via morphological gradient
6. Classical intensity quantification (OD values) → positivity %
7. AI classification (ResNet18) → border & inner intensity scores (0–3)
8. Overlay visualization with contours

### Key Modules

- **`src/main.py`** — Streamlit UI application entry point. Synchronous file upload and analysis pipeline with cached model loading. Teal theme.
- **`src/morphometry.py`** — `MorphometryAnalyzer` class: stain deconvolution (Ruifrok & Johnston matrices), tissue segmentation, positivity ratio calculation, zonal intensity analysis, AI inference, and overlay generation.
- **`src/model_utils.py`** — `LectinClassifier`: ResNet18 with two FC heads (border_score, inner_score, each 0–3). Inference only — training lives in a separate repository.

### Key Design Decisions

- **Hybrid approach**: classical morphometry for positivity %, deep learning for intensity classification
- **Multi-task learning**: single ResNet18 backbone with separate heads for border and inner scores
- **Zonal analysis**: border uses top 15% intensity (peak detection), inner uses all positive pixels (diffuse staining)
- **OD threshold at 0.25**: reduces noise in negative samples
- **Model file**: `src/assets/lectin_model.pth` (pre-trained weights, ~46MB)

## Dependencies

Defined in `pyproject.toml`. Core runtime requires `streamlit>=1.30.0`, `opencv-python`, `pillow`, `torchvision`, and `pandas`.
