#!/usr/bin/env python3
import argparse
import sys
import time

import cv2
import mediapipe as mp

sys.path.insert(0, "/home/pi/Transbot/py_install")
from Transbot_Lib import Transbot  # noqa: E402


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def main():
    parser = argparse.ArgumentParser(description="Track a hand with the Transbot camera gimbal.")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to run; 0 means forever.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--center-x", type=float, default=90.0)
    parser.add_argument("--center-y", type=float, default=115.0)
    parser.add_argument("--min-x", type=float, default=55.0)
    parser.add_argument("--max-x", type=float, default=125.0)
    parser.add_argument("--min-y", type=float, default=80.0)
    parser.add_argument("--max-y", type=float, default=145.0)
    parser.add_argument("--gain-x", type=float, default=0.028)
    parser.add_argument("--gain-y", type=float, default=0.026)
    parser.add_argument("--max-step", type=float, default=3.0)
    parser.add_argument("--send-interval", type=float, default=0.08)
    parser.add_argument("--invert-x", action="store_true")
    parser.add_argument("--invert-y", action="store_true")
    parser.add_argument("--no-recenter", action="store_true")
    args = parser.parse_args()

    bot = Transbot(debug=False)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.55,
    )

    servo_x = args.center_x
    servo_y = args.center_y
    last_send = 0.0
    seen_count = 0
    lost_count = 0
    start = time.monotonic()

    def send_angles(x, y):
        bot.set_pwm_servo(1, int(round(x)))
        time.sleep(0.01)
        bot.set_pwm_servo(2, int(round(y)))

    print("hand_gimbal_demo=start")
    print("Move your hand in front of the camera. Ctrl-C to stop.")
    send_angles(servo_x, servo_y)

    try:
        while True:
            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                print("camera_read=failed")
                time.sleep(0.1)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            now = time.monotonic()

            if result.multi_hand_landmarks:
                lost_count = 0
                seen_count += 1
                lm = result.multi_hand_landmarks[0].landmark
                cx = sum(point.x for point in lm) / len(lm)
                cy = sum(point.y for point in lm) / len(lm)
                err_x = cx - 0.5
                err_y = cy - 0.5

                sign_x = -1.0 if args.invert_x else 1.0
                sign_y = -1.0 if args.invert_y else 1.0
                step_x = clamp(sign_x * err_x * args.width * args.gain_x, -args.max_step, args.max_step)
                step_y = clamp(sign_y * err_y * args.height * args.gain_y, -args.max_step, args.max_step)

                servo_x = clamp(servo_x + step_x, args.min_x, args.max_x)
                servo_y = clamp(servo_y + step_y, args.min_y, args.max_y)

                if now - last_send >= args.send_interval:
                    send_angles(servo_x, servo_y)
                    last_send = now
                    if seen_count % 8 == 1:
                        print(
                            "hand center=({:.2f},{:.2f}) servo=({:.0f},{:.0f})".format(
                                cx, cy, servo_x, servo_y
                            )
                        )
            else:
                lost_count += 1
                if not args.no_recenter and lost_count > 18 and now - last_send >= args.send_interval:
                    servo_x += clamp(args.center_x - servo_x, -1.0, 1.0)
                    servo_y += clamp(args.center_y - servo_y, -1.0, 1.0)
                    send_angles(servo_x, servo_y)
                    last_send = now

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("hand_gimbal_demo=interrupted")
    finally:
        print("hand_gimbal_demo=centering")
        send_angles(args.center_x, args.center_y)
        cap.release()
        hands.close()
        del bot
        print("hand_gimbal_demo=done")


if __name__ == "__main__":
    main()
