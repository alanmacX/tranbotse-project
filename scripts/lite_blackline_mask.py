#!/usr/bin/env python3
import argparse
import time

import cv2 as cv
import numpy as np


def build_blackline_mask(frame, roi_top=0.55, min_area=250, crop=None):
    """Return annotated frame, binary mask, and line center info."""
    h, w = frame.shape[:2]
    if crop is None:
        x0, x1 = 0, w
        y0, y1 = int(h * roi_top), h
    else:
        x0, y0, x1, y1 = crop
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)

    roi = frame[y0:y1, x0:x1]
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # Extremely light black-line prior:
    # dark pixels in grayscale, plus low/medium V in HSV. Otsu adapts to lighting.
    _, otsu_dark = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    hsv_dark = cv.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 95]))
    mask = cv.bitwise_or(otsu_dark, hsv_dark)

    # Clean tiny noise and reconnect broken black paint edges.
    open_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    close_kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel, iterations=2)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) >= min_area]

    annotated = frame.copy()
    cv.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), (80, 180, 255), 2)

    if not contours:
        return annotated, mask, {"found": False, "cx": None, "cy": None, "error": None, "area": 0}

    contour = max(contours, key=cv.contourArea)
    area = float(cv.contourArea(contour))
    moments = cv.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        return annotated, mask, {"found": False, "cx": None, "cy": None, "error": None, "area": area}

    cx = int(moments["m10"] / moments["m00"]) + x0
    cy = int(moments["m01"] / moments["m00"]) + y0
    error = cx - (w // 2)

    shifted = contour + np.array([[[x0, y0]]])
    cv.drawContours(annotated, [shifted], -1, (0, 255, 0), 2)
    cv.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
    cv.line(annotated, (w // 2, y0), (w // 2, y1), (255, 0, 0), 1)
    cv.line(annotated, ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1), (255, 200, 0), 1)
    cv.line(annotated, (cx, cy), (w // 2, cy), (0, 0, 255), 2)
    cv.putText(
        annotated,
        f"cx={cx} err={error} area={int(area)}",
        (16, 32),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    return annotated, mask, {"found": True, "cx": cx, "cy": cy, "error": error, "area": area}


def main():
    parser = argparse.ArgumentParser(description="Lite black-line mask debugger.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", default="")
    parser.add_argument("--roi-top", type=float, default=0.55)
    parser.add_argument("--crop", default="", help="optional crop x0,y0,x1,y1")
    parser.add_argument("--min-area", type=int, default=250)
    parser.add_argument("--save-prefix", default="/tmp/blackline")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.image:
        frame = cv.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read image: {args.image}")
        crop = tuple(map(int, args.crop.split(","))) if args.crop else None
        annotated, mask, info = build_blackline_mask(frame, args.roi_top, args.min_area, crop)
        cv.imwrite(args.save_prefix + "_annotated.jpg", annotated)
        cv.imwrite(args.save_prefix + "_mask.png", mask)
        print(info)
        return

    cap = cv.VideoCapture(args.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = tuple(map(int, args.crop.split(","))) if args.crop else None
        annotated, mask, info = build_blackline_mask(frame, args.roi_top, args.min_area, crop)
        now = time.time()
        fps = 1.0 / max(now - last, 1e-6)
        last = now
        print(f"found={info['found']} error={info['error']} area={int(info['area'])} fps={fps:.1f}")

        if args.show:
            mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            cv.imshow("lite_blackline", np.hstack([annotated, mask_bgr]))
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            cv.imwrite(args.save_prefix + "_annotated.jpg", annotated)
            cv.imwrite(args.save_prefix + "_mask.png", mask)
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
