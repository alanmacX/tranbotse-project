#!/usr/bin/env python3
import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

TRANSBOT_LIB = "/home/pi/Transbot/py_install"
YOLO_DIR = "/home/pi/Software/yolov4-tiny-tf2"

sys.path.insert(0, TRANSBOT_LIB)
from Transbot_Lib import Transbot  # noqa: E402


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def load_yolo(score):
    os.chdir(YOLO_DIR)
    sys.path.insert(0, YOLO_DIR)
    from utils.yolo import YOLO  # noqa: E402

    return YOLO(score=score)


def detect_bottle(yolo, frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    annotated, boxes, scores, classes = yolo.detect_image(image)
    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        name = yolo.class_names[int(cls)]
        if name != "bottle":
            continue
        top, left, bottom, right = [float(x) for x in box]
        w = max(0.0, right - left)
        h = max(0.0, bottom - top)
        detections.append(
            {
                "score": float(score),
                "box": (top, left, bottom, right),
                "cx": (left + right) / 2.0,
                "cy": (top + bottom) / 2.0,
                "w": w,
                "h": h,
                "area": w * h,
            }
        )
    detections.sort(key=lambda d: d["score"] * d["area"], reverse=True)
    annotated_bgr = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
    return (detections[0] if detections else None), annotated_bgr


def stop(bot):
    for _ in range(3):
        bot.set_car_motion(0.0, 0.0)
        time.sleep(0.04)


def center_camera(bot):
    bot.set_pwm_servo(1, 90)
    time.sleep(0.08)
    bot.set_pwm_servo(2, 115)
    time.sleep(0.2)


def grab_bottle(bot, close_angle):
    close_angle = int(clamp(close_angle, 90, 150))
    print("grab=start close_angle={}".format(close_angle), flush=True)
    stop(bot)
    bot.set_uart_servo_angle_array(110, 180, 45, 1200)
    time.sleep(1.3)
    bot.set_uart_servo_angle_array(110, 180, close_angle, 900)
    time.sleep(1.1)
    bot.set_uart_servo_angle_array(110, 90, close_angle, 1200)
    time.sleep(1.3)
    print("grab=done holding", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Approach a YOLO-detected bottle and grab it.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--score", type=float, default=0.35)
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--max-pulses", type=int, default=10)
    parser.add_argument("--stop-height-ratio", type=float, default=0.46)
    parser.add_argument("--stop-bottom-ratio", type=float, default=0.82)
    parser.add_argument("--forward-speed", type=float, default=0.055)
    parser.add_argument("--forward-pulse", type=float, default=0.28)
    parser.add_argument("--turn-gain", type=float, default=0.75)
    parser.add_argument("--max-turn", type=float, default=0.22)
    parser.add_argument("--close-angle", type=int, default=130)
    parser.add_argument("--output", default="/tmp/yolo_bottle_demo.jpg")
    args = parser.parse_args()

    bot = Transbot(debug=False)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    try:
        stop(bot)
        center_camera(bot)
        yolo = load_yolo(args.score)
        print("demo=start detect_only={}".format(args.detect_only), flush=True)

        last_detection = None
        for pulse in range(args.max_pulses if not args.detect_only else 1):
            stop(bot)
            frame = None
            ok = False
            for _ in range(4):
                ok, frame = cap.read()
                time.sleep(0.04)
            if not ok or frame is None:
                print("camera_read=failed", flush=True)
                break

            detection, annotated = detect_bottle(yolo, frame)
            cv2.imwrite(args.output, annotated)
            if detection is None:
                print("pulse={} bottle=not_found stop".format(pulse), flush=True)
                break

            last_detection = detection
            h, w = frame.shape[:2]
            cx_ratio = detection["cx"] / w
            height_ratio = detection["h"] / h
            bottom_ratio = detection["box"][2] / h
            offset = cx_ratio - 0.5
            print(
                "pulse={} bottle score={:.2f} cx={:.2f} h={:.2f} bottom={:.2f}".format(
                    pulse, detection["score"], cx_ratio, height_ratio, bottom_ratio
                ),
                flush=True,
            )

            if args.detect_only:
                break

            if height_ratio >= args.stop_height_ratio or bottom_ratio >= args.stop_bottom_ratio:
                print("approach=stop_near_bottle", flush=True)
                stop(bot)
                grab_bottle(bot, args.close_angle)
                return

            angular = clamp(-offset * args.turn_gain, -args.max_turn, args.max_turn)
            speed = args.forward_speed if abs(offset) < 0.22 else 0.0
            print("motion pulse speed={:.3f} angular={:.3f}".format(speed, angular), flush=True)
            bot.set_car_motion(speed, angular)
            time.sleep(args.forward_pulse)

        stop(bot)
        if last_detection is None:
            print("demo=aborted no_bottle", flush=True)
        else:
            print("demo=ended_without_grab", flush=True)
    except KeyboardInterrupt:
        print("demo=interrupted", flush=True)
        stop(bot)
    finally:
        stop(bot)
        cap.release()
        del bot
        print("demo=done", flush=True)


if __name__ == "__main__":
    main()
