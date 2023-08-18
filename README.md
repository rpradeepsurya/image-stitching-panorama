
# Background Stitching and Image Panorama

## Overview

This project is part of the coursework on computer vision and image processing at UB and it is designed to experiment with image stitching methods, focusing on:

1. **Background Stitching**: Combining two images with similar backgrounds but differing foregrounds and removing moving foreground objects.
2. **Image Panorama**: Merging multiple images to create one panoramic photograph.

## Implementation Details

- Uses the **SIFT feature extractor** for detecting and matching keypoints between images.
- Employs **RANSAC** for optimizing the homography computation between overlapping image pairs.
- Saves overlap arrays in JSON format for further reference.

## Dependencies

To run the scripts, you need:

- **Python**: The code is written in Python.
- **OpenCV**, **Numpy**, and **Matplotlib**: These can be installed using:

```bash
pip install opencv-python numpy matplotlib
```

**Note**: Modules `json`, `itertools`, and `multiprocessing` are part of the Python standard library and don't require separate installation.

## How to Run

### 1. Background Stitching

**Steps**:
- Extract key points for each image.
- Extract features from each key point.
- Match features to determine overlaps.
- Compute the homography using RANSAC.
- Transform and stitch images, eliminating the foreground.

**Execution**:
1. Place your images in the `./images/` directory. By default, the script expects `t1_1.png` and `t1_2.png`.
2. Run:

```bash
python background_stitching.py
```

The stitched image will be saved as `task1.png`.

### 2. Image Panorama

**Steps**:
- Extract features for each image.
- Match features and determine spatial overlaps.
- Transform and stitch images into one panoramic photo.

**Execution**:
1. Ensure your images are in the correct directories. The script looks for images prefixed with `'t2'` and `'t3'` by default.
2. Run:

```bash
python panorama.py
```

Outputs will be stored as `task2.png` and `task3.png`. Overlap arrays will be saved in `t2_overlap.txt` and `t3_overlap.txt`.

---

