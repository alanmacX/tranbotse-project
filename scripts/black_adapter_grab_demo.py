#!/usr/bin/env python3
import argparse
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/pi/Transbot/py_install")
from Transbot_Lib import Transbot  # noqa: E402


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def stop(bot):
    for _ in range(3):
        bot.set_car_motion(0.0, 0.0)
        time.sleep(0.04)


def draw_label(frame, text, y=28):
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)


def find_black_adapter(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Dark, low-value objects. Keep this broad: black plastic varies a lot with glare.
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 115]))

    # Ignore borders and the top half of the frame to avoid walls, ceiling edges, and banners.
    roi = np.zeros_like(mask)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    y1, y2 = int(h * 0.45), int(h * 0.86)
    roi[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, roi)

    mask = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 350 or area > w * h * 0.08:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 24 or bh < 24:
            continue
        if bw > w * 0.22 or bh > h * 0.26:
            continue
        if x <= x1 + 2 or x + bw >= x2 - 2 or y <= y1 + 2 or y + bh >= y2 - 2:
            # Avoid objects cut off by the ROI edge; they are often false positives.
            continue
        aspect = bw / float(bh)
        if aspect < 0.35 or aspect > 3.2:
            continue
        rect_area = bw * bh
        fill = area / float(rect_area)
        if fill < 0.12:
            continue
        bottom_ratio = (y + bh) / float(h)
        center_bias = 1.0 - min(abs((x + bw / 2.0) / w - 0.5) * 1.8, 0.9)
        score = area * (bottom_ratio ** 2.0) * center_bias
        candidates.append((score, area, x, y, bw, bh, fill))

    if not candidates:
        return None, mask
    candidates.sort(reverse=True)
    _, area, x, y, bw, bh, fill = candidates[0]
    return {
        "x": x,
        "y": y,
        "w": bw,
        "h": bh,
        "area": area,
        "fill": fill,
        "cx": x + bw / 2.0,
        "cy": y + bh / 2.0,
        "bottom": y + bh,
    }, mask


def grab(bot, close_angle):
    close_angle = int(clamp(close_angle, 105, 150))
    print(f"grab=start close_angle={close_angle}", flush=True)
    stop(bot)
    bot.set_uart_servo_angle_array(110, 180, 45, 1200)
    time.sleep(1.3)
    bot.set_uart_servo_angle_array(110, 180, close_angle, 900)
    time.sleep(1.1)
    bot.set_uart_servo_angle_array(110, 90, close_angle, 1200)
    time.sleep(1.3)
    print("grab=done", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Approach and grab a black power adapter.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--max-pulses", type=int, default=10)
    parser.add_argument("--speed", type=float, default=0.045)
    parser.add_argument("--pulse", type=float, default=0.24)
    parser.add_argument("--turn-gain", type=float, default=0.75)
    parser.add_argument("--max-turn", type=float, default=0.18)
    parser.add_argument("--stop-height-ratio", type=float, default=0.42)
    parser.add_argument("--stop-bottom-ratio", type=float, default=0.84)
    parser.add_argument("--close-angle", type=int, default=132)
    parser.add_argument("--output", default="/tmp/black_adapter_bbox.jpg")
    args = parser.parse_args()

    bot = Transbot(debug=False)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    try:
        stop(bot)
        # Use the lowest pitch we tested; if the mount points upward, the preview will reveal that safely.
        bot.set_pwm_servo(1, 90)
        time.sleep(0.08)
        bot.set_pwm_servo(2, 0)
        time.sleep(0.4)

        for pulse_i in range(1 if args.preview_only else args.max_pulses):
            stop(bot)
            frame = None
            ok = False
            for _ in range(8):
                ok, frame = cap.read()
                time.sleep(0.035)
            if not ok or frame is None:
                print("camera=failed", flush=True)
                break

            target, mask = find_black_adapter(frame)
            annotated = frame.copy()
            if target is None:
                draw_label(annotated, "black adapter not detected")
                cv2.imwrite(args.output, annotated)
                print(f"pulse={pulse_i} target=not_found output={args.output}", flush=True)
                break

            h, w = frame.shape[:2]
            x, y, bw, bh = target["x"], target["y"], target["w"], target["h"]
            cx_ratio = target["cx"] / w
            height_ratio = bh / h
            bottom_ratio = target["bottom"] / h
            offset = cx_ratio - 0.5
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            draw_label(
                annotated,
                "black adapter? cx={:.2f} h={:.2f} bottom={:.2f}".format(
                    cx_ratio, height_ratio, bottom_ratio
                ),
            )
            cv2.imwrite(args.output, annotated)
            print(
                "pulse={} target=found cx={:.2f} h={:.2f} bottom={:.2f} area={:.0f}".format(
                    pulse_i, cx_ratio, height_ratio, bottom_ratio, target["area"]
                ),
                flush=True,
            )

            if args.preview_only:
                break

            if height_ratio >= args.stop_height_ratio or bottom_ratio >= args.stop_bottom_ratio:
                print("approach=near stop_and_grab", flush=True)
                stop(bot)
                grab(bot, args.close_angle)
                return

            angular = clamp(-offset * args.turn_gain, -args.max_turn, args.max_turn)
            speed = args.speed if abs(offset) <= 0.24 else 0.0
            print("motion speed={:.3f} angular={:.3f}".format(speed, angular), flush=True)
            bot.set_car_motion(speed, angular)
            time.sleep(args.pulse)

        stop(bot)
        print("demo=ended_without_grab", flush=True)
    except KeyboardInterrupt:
        print("demo=interrupted", flush=True)
    finally:
        stop(bot)
        cap.release()
        del bot
        print("demo=done", flush=True)


if __name__ == "__main__":
    main()
