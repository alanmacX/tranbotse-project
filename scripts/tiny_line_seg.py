#!/usr/bin/env python3
import argparse
import glob
import os
import time

import cv2 as cv
import numpy as np


IMG_W = 160
IMG_H = 120


def pseudo_blackline_mask(frame, roi_top=0.45):
    h, w = frame.shape[:2]
    y0 = int(h * roi_top)
    roi = frame[y0:h, :]
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    _, otsu_dark = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    hsv_dark = cv.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 95]))
    mask_roi = cv.bitwise_or(otsu_dark, hsv_dark)

    mask_roi = cv.morphologyEx(
        mask_roi,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    mask_roi = cv.morphologyEx(
        mask_roi,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_RECT, (9, 5)),
        iterations=2,
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:h, :] = mask_roi
    return mask


def build_model():
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(IMG_H, IMG_W, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    x = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.SeparableConv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.SeparableConv2D(24, 3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.SeparableConv2D(24, 3, strides=1, padding="same", activation="relu")(x)

    x = tf.keras.layers.UpSampling2D(2, interpolation="bilinear")(x)
    x = tf.keras.layers.SeparableConv2D(12, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D(2, interpolation="bilinear")(x)
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="tiny_line_seg")


def dice_loss(y_true, y_pred):
    import tensorflow as tf

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return 1.0 - tf.reduce_mean((2.0 * inter + 1.0) / (union + 1.0))


def loss_fn(y_true, y_pred):
    import tensorflow as tf

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


def load_dataset(data_dir):
    xs, ys = [], []
    for img_path in sorted(glob.glob(os.path.join(data_dir, "images", "*.jpg"))):
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(data_dir, "masks", name + ".png")
        if not os.path.exists(mask_path):
            continue
        img = cv.imread(img_path)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img = cv.resize(img, (IMG_W, IMG_H), interpolation=cv.INTER_AREA)
        mask = cv.resize(mask, (IMG_W, IMG_H), interpolation=cv.INTER_NEAREST)
        xs.append(img)
        ys.append((mask > 127).astype(np.float32)[..., None])
    if not xs:
        raise SystemExit("no training samples found")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def collect(args):
    os.makedirs(os.path.join(args.data, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.data, "masks"), exist_ok=True)
    cap = cv.VideoCapture(args.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    saved = 0
    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            continue
        if saved % max(args.stride, 1) == 0:
            mask = pseudo_blackline_mask(frame, args.roi_top)
            name = f"{int(time.time() * 1000)}_{saved:04d}"
            cv.imwrite(os.path.join(args.data, "images", name + ".jpg"), frame)
            cv.imwrite(os.path.join(args.data, "masks", name + ".png"), mask)
            print("saved", name)
        saved += 1
        time.sleep(max(args.delay, 0.0))
    cap.release()


def train(args):
    import tensorflow as tf

    x, y = load_dataset(args.data)
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=loss_fn)
    model.fit(x, y, batch_size=args.batch, epochs=args.epochs, validation_split=0.15, shuffle=True)
    model.save(args.model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(args.tflite, "wb") as f:
        f.write(tflite_model)
    print("saved", args.model)
    print("saved", args.tflite)
    model.summary()


def load_interpreter(tflite_path):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return interpreter, input_detail, output_detail


def infer_one(frame, interpreter, input_detail, output_detail, threshold=0.5):
    small = cv.resize(frame, (IMG_W, IMG_H), interpolation=cv.INTER_AREA).astype(np.float32)
    inp = np.expand_dims(small, axis=0)
    interpreter.set_tensor(input_detail["index"], inp)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_detail["index"])[0, :, :, 0]
    mask_small = (prob > threshold).astype(np.uint8) * 255
    mask = cv.resize(mask_small, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_NEAREST)

    h, w = mask.shape[:2]
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) >= 250]
    info = {"found": False, "cx": None, "error": None, "area": 0}
    annotated = frame.copy()
    if contours:
        c = max(contours, key=cv.contourArea)
        m = cv.moments(c)
        area = float(cv.contourArea(c))
        if abs(m["m00"]) > 1e-6:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            info = {"found": True, "cx": cx, "error": cx - w // 2, "area": area}
            cv.drawContours(annotated, [c], -1, (0, 255, 0), 2)
            cv.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            cv.putText(annotated, f"err={info['error']} area={int(area)}", (16, 32),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return annotated, mask, info


def infer(args):
    interpreter, input_detail, output_detail = load_interpreter(args.tflite)
    cap = cv.VideoCapture(args.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        annotated, mask, info = infer_one(frame, interpreter, input_detail, output_detail, args.threshold)
        now = time.time()
        fps = 1.0 / max(now - last, 1e-6)
        last = now
        print(f"found={info['found']} error={info['error']} area={int(info['area'])} fps={fps:.1f}")
        if args.show:
            cv.imshow("tiny_line_seg", np.hstack([annotated, cv.cvtColor(mask, cv.COLOR_GRAY2BGR)]))
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            cv.imwrite(args.out_prefix + "_annotated.jpg", annotated)
            cv.imwrite(args.out_prefix + "_mask.png", mask)
            break
    cap.release()
    cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Tiny deep-learning line segmentation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--data", default="/home/pi/line_seg_data")
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--delay", type=float, default=0.03)
    p.add_argument("--roi-top", type=float, default=0.45)
    p.set_defaults(func=collect)

    p = sub.add_parser("train")
    p.add_argument("--data", default="/home/pi/line_seg_data")
    p.add_argument("--model", default="/home/pi/tiny_line_seg.keras")
    p.add_argument("--tflite", default="/home/pi/tiny_line_seg.tflite")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.set_defaults(func=train)

    p = sub.add_parser("infer")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--tflite", default="/home/pi/tiny_line_seg.tflite")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out-prefix", default="/home/pi/tiny_line_seg")
    p.add_argument("--show", action="store_true")
    p.set_defaults(func=infer)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
