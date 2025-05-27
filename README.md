# Achilles Reflex Tracking – Summer 2025

## Overview

This project reconstructs a pipeline for video-based measurement of Achilles tendon reflex responses, with the goal of automatically tracking reflex movement, removing foot drift, and detecting reflex onset and filtering valid reflex complexes. The data is based on reflex recordings from healthy subjects collected under Dr. Podcheko.

## Current To-Do

- [ ] Remove trend from vertical displacement signal (baseline correction) [IN PROGRESS]
- [ ] Implement half relaxation time calculation algorithm
- [ ] Detect reflex onset from the displacement curve
- [ ] Determine which reflex complexes (from multi-tap sequences) are valid and filter out noisy/faulty ones
- [ ] Move to Jupyter notebook? 
- [ ] MVP fullstack implementation for web

## Progress Log

**16 May** – Initial work on marker detection. Compared Hough Circles vs. blob detection. Blob proved more stable; noted possible extension to ellipse detection for greater generality.  
**22 May** – Implemented tracking for graph generation; tested across multiple videos.  
**26 May** – Began testing baseline/trend removal methods. Referenced time series text, p.21, for trend modeling techniques. Graphs generated based on all videos provided

## Notes

Further steps include:
- annotating frames for validation
- adapting the pipeline for variable-quality input data.

## Project Structure

```
thyroid/
├── README.md
├── main.py          # Current implementation of reflex tracking and analysis
└── graphs/          # Generated PNG visualizations from video tracking
    └── *.png        # Various graph outputs from analysis
```
