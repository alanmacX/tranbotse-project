#!/usr/bin/env python3
"""
calibrate_perspective.py - Interactive IPM / Bird's Eye View Calibrator

This script allows you to generate the 'homography' matrix needed to map
camera pixels to real-world coordinates, solving the perspective distortion
problem for pure-pursuit path tracking.

Instructions:
1. Place the robot on the ground.
2. Place a standard A4 paper (210mm x 297mm) centrally in front of the robot.
   Measure the ground distance from the camera's vertical projection to the
   paper's near edge and pass it as --paper-near-cm.
3. Run this script.
4. Click the 4 corners of the A4 paper in the image IN THIS EXACT ORDER:
   - Bottom-Left
   - Top-Left
   - Top-Right
   - Bottom-Right
5. Press 'c' to calculate and preview the warped image.
6. Press 's' to save the calibration to configs/race_config.json and exit.
7. Press 'r' to reset points, or 'q' to quit without saving.
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2 as cv
from pathlib import Path

# Add project root to sys.path so we can import modules if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

CONFIG_PATH = Path('configs/race_config.json')

# Real-world dimensions of the calibration object (A4 paper)
# A4 is 21cm x 29.7cm. We use a scale of 10 pixels per cm for the destination image.
# You can change these if you use a different calibration square.
PAPER_W_CM = 21.0
PAPER_H_CM = 29.7
PX_PER_CM = 4.0  # Must match perspective.px_per_cm in config

DST_W = int(PAPER_W_CM * PX_PER_CM)
DST_H = int(PAPER_H_CM * PX_PER_CM)

# We want the paper to appear in the center-bottom of our 160x240 output BEV image
OUT_W = 160
OUT_H = 240
# Horizontal offset; the vertical offset is computed from --paper-near-cm.
OFFSET_X = (OUT_W - DST_W) // 2

clicked_pts = []

def mouse_callback(event, x, y, flags, param):
    global clicked_pts
    if event == cv.EVENT_LBUTTONDOWN:
        if len(clicked_pts) < 4:
            clicked_pts.append((x, y))
            print(f"Point {len(clicked_pts)} recorded: ({x}, {y})")

def main():
    global clicked_pts

    parser = argparse.ArgumentParser(description="Calibrate Inverse Perspective Mapping (IPM)")
    parser.add_argument("--camera", default=0, help="Camera index or stream URL (default: 0)")
    parser.add_argument("--paper-near-cm", type=float, default=5.0,
                        help="Ground distance from camera projection to A4 near edge")
    parser.add_argument("image", nargs="?", help="Optional static image file instead of camera")
    args = parser.parse_args()
    near_px = int(round(max(0.0, args.paper_near_cm) * PX_PER_CM))
    offset_y = OUT_H - DST_H - near_px
    if offset_y < 0:
        raise ValueError("paper-near-cm places the A4 paper outside the BEV output")
    dst_pts = np.float32([
        [OFFSET_X, offset_y + DST_H],
        [OFFSET_X, offset_y],
        [OFFSET_X + DST_W, offset_y],
        [OFFSET_X + DST_W, offset_y + DST_H],
    ])

    # Load camera config to get resolution
    if not CONFIG_PATH.exists():
        print(f"Error: {CONFIG_PATH} not found.")
        return

    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)

    cam_cfg = cfg.get("camera", {})
    w, h = cam_cfg.get("frame_width", 640), cam_cfg.get("frame_height", 480)

    if args.image:
        frame = cv.imread(args.image)
        if frame is None:
            print(f"Failed to read image: {args.image}")
            return
    else:
        # Try to parse camera arg as int if possible, otherwise use as string (URL)
        try:
            cam_source = int(args.camera)
        except ValueError:
            cam_source = args.camera

        cap = cv.VideoCapture(cam_source)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

        if not cap.isOpened():
            print(f"Failed to open camera: {cam_source}")
            print("Provide a static image path as argument if testing: ./scripts/calibrate_perspective.py [image_path]")
            return

        # Warm up
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture frame from camera.")
            return

    # Undistort if configured
    p_cfg = cfg.get("perspective", {})
    if p_cfg.get("undistort") and p_cfg.get("camera_matrix"):
        mtx = np.array(p_cfg["camera_matrix"]).reshape(3, 3)
        dist = np.array(p_cfg["dist_coeffs"])
        frame = cv.undistort(frame, mtx, dist)

    # Apply crop to match what the pipeline sees
    crop_cfg = cam_cfg.get("crop", [300, 265, 430, 455])
    cx0, cy0, cx1, cy1 = crop_cfg
    ex_l = cam_cfg.get("expand_left_px", 20)
    ex_r = cam_cfg.get("expand_right_px", 140)

    x0e = max(0, cx0 - ex_l)
    x1e = min(w, cx1 + ex_r)

    crop = frame[cy0:cy1, x0e:x1e].copy()
    display_crop = crop.copy()

    cv.namedWindow("Calibration - Click 4 corners (BL, TL, TR, BR)")
    cv.setMouseCallback("Calibration - Click 4 corners (BL, TL, TR, BR)", mouse_callback)

    print(__doc__)

    homography = None

    while True:
        disp = display_crop.copy()
        for i, pt in enumerate(clicked_pts):
            cv.circle(disp, pt, 3, (0, 0, 255), -1)
            cv.putText(disp, str(i+1), (pt[0]+5, pt[1]-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if i > 0:
                cv.line(disp, clicked_pts[i-1], pt, (255, 0, 0), 1)
        if len(clicked_pts) == 4:
            cv.line(disp, clicked_pts[3], clicked_pts[0], (255, 0, 0), 1)

        cv.imshow("Calibration - Click 4 corners (BL, TL, TR, BR)", disp)
        key = cv.waitKey(50) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            clicked_pts.clear()
            homography = None
            print("Points reset.")
        elif key == ord('c'):
            if len(clicked_pts) == 4:
                src_pts = np.float32(clicked_pts)
                homography = cv.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv.warpPerspective(crop, homography, (OUT_W, OUT_H))
                cv.imshow("Bird's Eye View Preview", warped)
                print("Calculated Homography Matrix. Press 's' to save, 'r' to reset, or click window and press 'q' to quit.")
            else:
                print("Please click exactly 4 points first.")
        elif key == ord('s'):
            if homography is not None:
                # Update config
                p_cfg["enabled"] = True
                p_cfg["homography"] = homography.flatten().tolist()
                p_cfg["output_width"] = OUT_W
                p_cfg["output_height"] = OUT_H
                cfg["perspective"] = p_cfg

                with open(CONFIG_PATH, 'w') as f:
                    json.dump(cfg, f, indent=2)
                print(f"Calibration saved successfully to {CONFIG_PATH}!")
                print("Perspective mode is now ENABLED. Restart the race runner to apply.")
                break
            else:
                print("Calculate homography first by pressing 'c'.")

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
