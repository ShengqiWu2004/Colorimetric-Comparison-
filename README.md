# Colorimetric-Comparison

This repository contains code for evaluating and comparing color correction methods used in smartphone-based colorimetric analysis of chemical solutions.

#### 📂 Structure

- `color_process.py`: Main script to process solution images and apply three color correction methods.
- `model.py`: Defines the deep learning-based ensemble model used for concentration prediction, integrating both visual and colorimetric features.
- `models/`: Folder containing two pre-trained models used for the comparison described in the accompanying manuscript.
- `list/`: position for putting in test images

#### 🧪 Correction Methods

Three approaches are implemented and compared:

1. **Alpha Blending** – Classic transparency correction using known background and estimated alpha values.
2. **Beer-Lambert Law** – Physics-inspired transmittance model.
3. **Linear Regression** – A learned model trained on observed RGB and background values.

These are evaluated using color swatches and plotted visualizations (`color_grid.png`) to demonstrate method differences.

#### 🧠 Deep Learning Model

The ensemble model combines:

- A lightweight CNN extracting spatial features from input images.
- Engineered color and lighting features (e.g., contrast, uniformity).
- Fusion layers for concentration prediction.

Detailed implementation and training strategy are in `model.py`.