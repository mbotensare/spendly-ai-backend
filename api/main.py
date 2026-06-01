"""
Spendly AI — FastAPI Inference Server v2.1
===========================================
REST API untuk serving 4 model ML Spendly:
  - Receipt Detector : deteksi region date/store/total (MiniYOLO Dual-Scale)
  - OCR CRNN         : ekstraksi teks dari crop region
  - Classifier       : klasifikasi 9 kategori pengeluaran (TF-IDF + CNN + SE)
  - Forecaster       : prediksi pengeluaran mingguan (LSTM + Attention)

Endpoint tambahan:
  - /insight          : saran keuangan via Google Gemini Generative AI
  - /process-receipt  : full pipeline (Detect regions → OCR per region → Classify)
  - /detect-receipt   : deteksi bounding box date/store/total dalam foto
  - /health           : status server

Jalankan:
  uvicorn api.main:app --reload --port 8000

Coding Camp 2026 powered by DBS Foundation (CC26-PSU276)
"""

import os
import re
import sys
import string
import io
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spendly-api")

# ── Path Setup ────────────────────────────────────────────────────────────────
API_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_DIR)
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ── Konstanta Global ──────────────────────────────────────────────────────────
CATEGORIES = [
    "Beauty", "F&B", "Gas", "Groceries", "Health",
    "HouseHold", "Lifestyle", "Listrik", "Transport",
]
LABELS           = CATEGORIES
LABEL_TO_IDX     = {lbl: i for i, lbl in enumerate(LABELS)}
NUM_CLASSES      = len(LABELS)
IMG_SIZE         = 224
LOOKBACK         = 12
N_FORECAST_FEATURES = NUM_CLASSES

BLUR_THRESHOLD   = 100

CHARS_OCR        = string.digits + string.ascii_uppercase + string.ascii_lowercase + " .,:-/()%"
BLANK_IDX        = len(CHARS_OCR) + 1
NUM_OCR_CHARS    = len(CHARS_OCR) + 2
OCR_IMG_W        = 400
OCR_IMG_H        = 50

CLASS_NAMES_DETECTOR = ["date", "store", "total"]  # NB07 3-class detector


# ══════════════════════════════════════════════════════════════════════════════
# INLINE CUSTOM KERAS COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    logger.warning("PySastrawi tidak tersedia. NLP preprocessing akan di-skip.")


class NLPPreprocessor:
    def __init__(self):
        if SASTRAWI_AVAILABLE:
            self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
            self.stemmer = StemmerFactory().create_stemmer()
        else:
            self.stopword_remover = None
            self.stemmer = None

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\d+", "NUM", text)
        text = re.sub(r"\s+", " ", text).strip()
        if self.stopword_remover:
            text = self.stopword_remover.remove(text)
        if self.stemmer:
            text = self.stemmer.stem(text)
        return text

    def preprocess_batch(self, texts):
        return [self.preprocess(t) for t in texts]


# ── Classifier Components ─────────────────────────────────────────────────────

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap     = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1  = tf.keras.layers.Dense(
            max(1, channels // self.ratio), activation="relu", use_bias=False
        )
        self.dense2  = tf.keras.layers.Dense(channels, activation="sigmoid", use_bias=False)
        self.reshape = tf.keras.layers.Reshape((1, 1, channels))
        super().build(input_shape)

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, inputs):
        score            = self.V(tf.nn.tanh(self.W(inputs)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector   = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma        = gamma
        self.alpha        = alpha
        self.class_weights = class_weights

        if class_weights is not None:
            vals  = list(class_weights.values())
            max_w = max(vals)
            self._alpha_tensor = tf.constant(
                [class_weights.get(i, 1.0) / max_w for i in range(len(class_weights))],
                dtype=tf.float32,
            )
        elif isinstance(alpha, (list, tuple)):
            self._alpha_tensor = tf.constant(alpha, dtype=tf.float32)
        else:
            self._alpha_tensor = float(alpha)

    def call(self, y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce      = -y_true * tf.math.log(y_pred)
        pt      = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal   = tf.math.pow(1.0 - pt, self.gamma) * ce

        if isinstance(self._alpha_tensor, float):
            weighted = self._alpha_tensor * focal
        else:
            alpha_t  = tf.reshape(self._alpha_tensor, (1, -1))
            weighted = alpha_t * y_true * focal

        return tf.reduce_mean(tf.reduce_sum(weighted, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma":         self.gamma,
            "alpha":         self.alpha,
            "class_weights": self.class_weights,
        })
        return config


# ── NB07 Detector Components ──────────────────────────────────────────────────

class L2NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, n_channels, scale_init=20.0, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.scale_init = scale_init

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, self.n_channels),
            initializer=tf.keras.initializers.Constant(self.scale_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        norm = tf.norm(x, axis=-1, keepdims=True) + 1e-7
        return self.gamma * (x / norm)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_channels": self.n_channels, "scale_init": self.scale_init})
        return cfg


class AnchorBoxLayer(tf.keras.layers.Layer):
    def __init__(self, anchors, grid_h, grid_w, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.anchors_list = anchors if isinstance(anchors, list) else anchors.tolist()
        self.grid_h       = grid_h
        self.grid_w       = grid_w
        self.num_classes  = num_classes
        self.num_anchors  = len(self.anchors_list)

    def build(self, input_shape):
        self._anchors_tf = tf.constant(self.anchors_list, dtype=tf.float32)
        super().build(input_shape)

    def call(self, raw):
        raw   = tf.cast(raw, tf.float32)
        batch = tf.shape(raw)[0]
        raw   = tf.reshape(raw, [batch, self.grid_h, self.grid_w,
                                  self.num_anchors, 5 + self.num_classes])

        grid_y = tf.range(self.grid_h, dtype=tf.float32)
        grid_x = tf.range(self.grid_w, dtype=tf.float32)
        gx, gy = tf.meshgrid(grid_x, grid_y)
        grid   = tf.stack([gx, gy], axis=-1)
        grid   = tf.reshape(grid, [1, self.grid_h, self.grid_w, 1, 2])

        xy      = (tf.sigmoid(raw[..., 1:3]) + grid)
        xy      = xy / tf.cast([self.grid_w, self.grid_h], tf.float32)
        anchors = tf.reshape(self._anchors_tf, [1, 1, 1, self.num_anchors, 2])
        wh      = tf.exp(tf.clip_by_value(raw[..., 3:5], -4.0, 4.0)) * anchors
        obj     = tf.sigmoid(raw[..., 0:1])
        cls     = tf.sigmoid(raw[..., 5:])

        return tf.concat([obj, xy, wh, cls], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "anchors":     self.anchors_list,
            "grid_h":      self.grid_h,
            "grid_w":      self.grid_w,
            "num_classes": self.num_classes,
        })
        return cfg


class DetectionLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_coord=10.0, lambda_noobj=0.3,
                 lambda_cls=1.0, num_classes=3,
                 label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_coord    = lambda_coord
        self.lambda_noobj    = lambda_noobj
        self.lambda_cls      = lambda_cls
        self.num_classes     = num_classes
        self.label_smoothing = label_smoothing

    @staticmethod
    def _ciou(box_true, box_pred):
        eps  = 1e-7
        tx1  = box_true[..., 0] - box_true[..., 2] / 2
        ty1  = box_true[..., 1] - box_true[..., 3] / 2
        tx2  = box_true[..., 0] + box_true[..., 2] / 2
        ty2  = box_true[..., 1] + box_true[..., 3] / 2
        px1  = box_pred[..., 0] - box_pred[..., 2] / 2
        py1  = box_pred[..., 1] - box_pred[..., 3] / 2
        px2  = box_pred[..., 0] + box_pred[..., 2] / 2
        py2  = box_pred[..., 1] + box_pred[..., 3] / 2

        inter_w = tf.maximum(tf.minimum(tx2, px2) - tf.maximum(tx1, px1), 0.0)
        inter_h = tf.maximum(tf.minimum(ty2, py2) - tf.maximum(ty1, py1), 0.0)
        inter   = inter_w * inter_h
        area_t  = box_true[..., 2] * box_true[..., 3]
        area_p  = box_pred[..., 2] * box_pred[..., 3]
        union   = area_t + area_p - inter + eps
        iou     = inter / union

        enc_x1  = tf.minimum(tx1, px1)
        enc_y1  = tf.minimum(ty1, py1)
        enc_x2  = tf.maximum(tx2, px2)
        enc_y2  = tf.maximum(ty2, py2)
        c2      = tf.square(enc_x2 - enc_x1) + tf.square(enc_y2 - enc_y1) + eps
        d2      = (tf.square(box_true[..., 0] - box_pred[..., 0]) +
                   tf.square(box_true[..., 1] - box_pred[..., 1]))

        atan_t  = tf.atan(box_true[..., 2] / (box_true[..., 3] + eps))
        atan_p  = tf.atan(box_pred[..., 2] / (box_pred[..., 3] + eps))
        v       = (4.0 / (np.pi ** 2)) * tf.square(atan_t - atan_p)
        alpha   = tf.stop_gradient(v / (1.0 - iou + v + eps))

        return 1.0 - (iou - d2 / c2 - alpha * v)

    def call(self, y_true, y_pred):
        obj_mask   = y_true[..., 0:1]
        noobj_mask = 1.0 - obj_mask
        eps        = 1e-7

        obj_pred  = y_pred[..., 0:1]
        obj_bce   = -(
            obj_mask   * tf.math.log(obj_pred + eps) +
            noobj_mask * tf.math.log(1.0 - obj_pred + eps)
        )
        focal_w   = tf.pow(tf.abs(obj_mask - obj_pred), 2.0)
        obj_bce   = focal_w * obj_bce

        cls_idx   = tf.argmax(y_true[..., 5:], axis=-1, output_type=tf.int32)
        is_total  = tf.cast(tf.equal(cls_idx, 2), tf.float32)
        is_total  = tf.expand_dims(is_total, axis=-1)
        obj_cls_w = 1.0 + 1.0 * is_total

        obj_loss  = (
            tf.reduce_mean(obj_cls_w * obj_mask   * obj_bce) +
            self.lambda_noobj * tf.reduce_mean(noobj_mask * obj_bce)
        )

        ciou_val   = self._ciou(y_true[..., 1:5], y_pred[..., 1:5])
        coord_loss = self.lambda_coord * tf.reduce_mean(
            obj_mask * tf.expand_dims(ciou_val, axis=-1)
        )

        cls_true        = y_true[..., 5:]
        cls_pred        = y_pred[..., 5:]
        smooth          = self.label_smoothing
        n               = tf.cast(self.num_classes, tf.float32)
        cls_true_smooth = cls_true * (1.0 - smooth) + smooth / n
        cls_bce         = -(
            cls_true_smooth * tf.math.log(cls_pred + eps) +
            (1.0 - cls_true_smooth) * tf.math.log(1.0 - cls_pred + eps)
        )
        class_weights = tf.constant([1.0, 1.0, 3.0], dtype=tf.float32)
        cls_bce       = cls_bce * class_weights
        cls_loss      = self.lambda_cls * tf.reduce_mean(
            obj_mask * tf.reduce_sum(cls_bce, axis=-1, keepdims=True)
        )

        return obj_loss + coord_loss + cls_loss

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "lambda_coord":    self.lambda_coord,
            "lambda_noobj":    self.lambda_noobj,
            "lambda_cls":      self.lambda_cls,
            "num_classes":     self.num_classes,
            "label_smoothing": self.label_smoothing,
        })
        return cfg


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_init, lr_min, warmup_steps, total_steps):
        self.lr_init           = lr_init
        self.lr_min            = lr_min
        self._warmup_steps_int = int(warmup_steps)
        self._total_steps_int  = int(total_steps)
        self.warmup_steps      = tf.cast(warmup_steps, tf.float32)
        self.total_steps       = tf.cast(total_steps, tf.float32)

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.lr_min + (self.lr_init - self.lr_min) * (
            step / self.warmup_steps
        )
        cosine_lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
            1.0 + tf.cos(
                np.pi * (step - self.warmup_steps) /
                (self.total_steps - self.warmup_steps + 1e-7)
            )
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "lr_init":      self.lr_init,
            "lr_min":       self.lr_min,
            "warmup_steps": self._warmup_steps_int,
            "total_steps":  self._total_steps_int,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_forecaster(lookback: int = LOOKBACK, n_features: int = N_FORECAST_FEATURES):
    inputs  = tf.keras.Input(shape=(lookback, n_features))
    x       = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x       = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x       = AttentionLayer(name="attention")(x)
    x       = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_features)(x)
    return tf.keras.Model(inputs, outputs, name="spendly_forecaster_attention")


def _build_ocr_model():
    from tensorflow.keras import layers

    inputs = tf.keras.Input(shape=(OCR_IMG_H, OCR_IMG_W, 1), name="ocr_input")
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.3)(x)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((-1, 6 * 128))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_OCR_CHARS, activation="linear", name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="spendly_ocr_crnn")


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def assess_blur(image_bgr: np.ndarray) -> dict:
    gray       = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {"blur_score": blur_score, "is_blurry": blur_score < BLUR_THRESHOLD}


def preprocess_image_classifier(image_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype(np.float32) / 255.0


def preprocess_image_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocessing identik dengan preprocess_real_crop() di NB05.
    Pipeline: CLAHE ringan → Denoise ringan → Resize LANCZOS4 → Linear histogram stretch P5-P95
    """
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()

    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)

    # 1. CLAHE ringan (identik NB05 preprocess_real_crop)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    # 2. Denoise ringan (identik NB05)
    gray = cv2.fastNlMeansDenoising(gray, None, h=5,
                                     templateWindowSize=7,
                                     searchWindowSize=15)

    # 3. Resize LANCZOS4 (identik NB05)
    gray = cv2.resize(gray, (OCR_IMG_W, OCR_IMG_H),
                      interpolation=cv2.INTER_LANCZOS4)

    # 4. Linear histogram stretch P5-P95 (identik NB05)
    p_low  = float(np.percentile(gray,  5))
    p_high = float(np.percentile(gray, 95))
    if p_high - p_low > 10:
        arr = (gray.astype(np.float32) - p_low) / (p_high - p_low)
        arr = np.clip(arr, 0.0, 1.0)
    else:
        arr = gray.astype(np.float32) / 255.0

    return arr


def preprocess_image_detector(image_bgr: np.ndarray, target_size: tuple) -> np.ndarray:
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[1], target_size[0]))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def decode_ocr(logits: np.ndarray) -> str:
    logits      = tf.cast(logits, tf.float32)
    pred_indices = tf.argmax(logits, axis=-1).numpy()[0]
    text        = ""
    prev_idx    = -1
    for idx in pred_indices:
        if idx != prev_idx:
            if 1 <= idx <= len(CHARS_OCR):
                text += CHARS_OCR[idx - 1]
        prev_idx = idx
    return text


def bytes_to_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak dapat dibaca. Pastikan format JPG/PNG.")
    return img


# ── Detector Decode & Crop ────────────────────────────────────────────────────

def _det_iou(a, b):
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    inter    = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union    = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-7
    return inter / union


def decode_detector_output(predictions, conf_thresh=0.35, iou_thresh=0.45):
    """
    Decode dual-scale MiniYOLO NB07 output.
    Input: list of 2 arrays [pred26, pred52], masing-masing shape
           (batch, grid_h, grid_w, num_anchors, 5+3) — sudah di-decode AnchorBoxLayer.
    Output: dict {"date": {...}, "store": {...}, "total": {...}}
    Format cell: [obj, cx, cy, w, h, cls_date, cls_store, cls_total]
    """

    def decode_one_head(pred, conf_thresh):
        detections = []
        if pred.ndim == 5:
            pred = pred[0]  # hilangkan batch → (gh, gw, na, 5+nc)
        if pred.ndim != 4:
            return detections
        gh, gw, na, depth = pred.shape
        for gy in range(gh):
            for gx in range(gw):
                for a in range(na):
                    cell = pred[gy, gx, a]
                    conf = float(cell[0])
                    if conf < conf_thresh:
                        continue
                    cx       = float(np.clip(cell[1], 0, 1))
                    cy       = float(np.clip(cell[2], 0, 1))
                    w        = float(np.clip(cell[3], 0, 1))
                    h        = float(np.clip(cell[4], 0, 1))
                    cls_prob = cell[5:]
                    cls_id   = int(np.argmax(cls_prob))
                    score    = conf * float(cls_prob[cls_id])
                    if score >= conf_thresh:
                        detections.append([cx, cy, w, h, score, cls_id])
        return detections

    def nms(detections, iou_thresh):
        detections.sort(key=lambda x: -x[4])
        kept = []
        while detections:
            best = detections.pop(0)
            kept.append(best)
            detections = [
                d for d in detections
                if _det_iou(best[:4], d[:4]) < iou_thresh or d[5] != best[5]
            ]
        # heuristik: kalau >1 total, ambil yang cy terbesar (paling bawah)
        total_dets = [d for d in kept if d[5] == 2]
        other_dets = [d for d in kept if d[5] != 2]
        if len(total_dets) > 1:
            total_dets = [max(total_dets, key=lambda x: x[1])]
        return other_dets + total_dets

    if isinstance(predictions, (list, tuple)) and len(predictions) >= 2:
        dets = (decode_one_head(np.array(predictions[0]), conf_thresh) +
                decode_one_head(np.array(predictions[1]), conf_thresh))
    else:
        pred = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        dets = decode_one_head(np.array(pred), conf_thresh)

    dets = nms(dets, iou_thresh)

    result = {}
    for d in dets:
        cx, cy, w, h, score, cls_id = d
        cls_name = CLASS_NAMES_DETECTOR[cls_id]
        if cls_name not in result or score > result[cls_name]["confidence"]:
            result[cls_name] = {
                "cx": cx, "cy": cy, "w": w, "h": h,
                "confidence": round(score, 4),
            }
    return result


def crop_region(img_bgr: np.ndarray, det: dict, padding: float = 0.03) -> np.ndarray:
    """Crop area gambar berdasarkan normalized cx,cy,w,h dari detector."""
    h, w  = img_bgr.shape[:2]
    cx, cy, bw, bh = det["cx"], det["cy"], det["w"], det["h"]
    x1 = max(0, int((cx - bw / 2 - padding) * w))
    y1 = max(0, int((cy - bh / 2 - padding) * h))
    x2 = min(w, int((cx + bw / 2 + padding) * w))
    y2 = min(h, int((cy + bh / 2 + padding) * h))
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]


def ocr_region(img_bgr_crop: np.ndarray) -> str:
    """OCR satu crop region. Return string."""
    if img_bgr_crop is None or "ocr" not in models:
        return ""
    try:
        img_ocr    = preprocess_image_ocr(img_bgr_crop)
        img_tensor = np.expand_dims(img_ocr, axis=(0, -1))
        logits     = model_predict(models["ocr"], img_tensor)
        return decode_ocr(logits)
    except Exception as e:
        logger.warning(f"OCR region gagal: {e}")
        return ""


# ── Generic Model Predict ─────────────────────────────────────────────────────

def model_predict(model, inputs):
    if hasattr(model, 'predict'):
        return model.predict(inputs, verbose=0)
    elif isinstance(model, tf.keras.layers.Layer):
        if isinstance(inputs, (list, tuple)):
            tensor_inputs = [
                tf.constant(x, dtype=tf.float32) if not isinstance(x, tf.Tensor) else x
                for x in inputs
            ]
            input_keys = None
            endpoint_fn = None
            for attr in ('_callable', '_endpoint'):
                fn = getattr(model, attr, None)
                if fn is not None and hasattr(fn, 'structured_input_signature'):
                    input_keys = list(fn.structured_input_signature[1].keys())
                    break
            if input_keys is None:
                saved_path = getattr(model, '_asset_path', None) or getattr(model, 'filepath', None)
                if saved_path:
                    sm = tf.saved_model.load(str(saved_path))
                    sig = sm.signatures.get('serving_default')
                    if sig:
                        input_keys  = list(sig.structured_input_signature[1].keys())
                        endpoint_fn = sig
            if input_keys and len(input_keys) >= len(tensor_inputs):
                kwargs = {input_keys[i]: tensor_inputs[i] for i in range(len(tensor_inputs))}
                result = endpoint_fn(**kwargs) if endpoint_fn else model(**kwargs)
            else:
                result = model(tensor_inputs[0],
                               tensor_inputs[1] if len(tensor_inputs) > 1 else None)
        else:
            input_tensor = (tf.constant(inputs, dtype=tf.float32)
                            if not isinstance(inputs, tf.Tensor) else inputs)
            result = model(input_tensor)

        if isinstance(result, dict):
            val = next(iter(result.values()))
            return val.numpy() if hasattr(val, 'numpy') else np.array(val)
        return result.numpy() if hasattr(result, 'numpy') else np.array(result)

    elif hasattr(model, 'signatures'):
        infer      = model.signatures['serving_default']
        input_keys = list(infer.structured_input_signature[1].keys())
        if isinstance(inputs, (list, tuple)):
            kwargs = {input_keys[i]: tf.constant(inputs[i], dtype=tf.float32)
                      for i in range(min(len(input_keys), len(inputs)))}
        else:
            kwargs = {input_keys[0]: tf.constant(inputs, dtype=tf.float32)}
        result = infer(**kwargs)
        val    = next(iter(result.values()))
        return val.numpy()
    else:
        if isinstance(inputs, (list, tuple)):
            tensors = [tf.constant(x, dtype=tf.float32) for x in inputs]
            result  = model(*tensors)
        else:
            result = model(tf.constant(inputs, dtype=tf.float32))
        return result.numpy() if hasattr(result, 'numpy') else np.array(result)


def model_predict_yolo(model, inputs):
    """Khusus MiniYOLO — handle dual-scale output (list of arrays)."""
    if hasattr(model, 'predict'):
        result = model.predict(inputs, verbose=0)
        return result
    elif hasattr(model, 'signatures'):
        infer      = model.signatures['serving_default']
        input_keys = list(infer.structured_input_signature[1].keys())
        kwargs     = {input_keys[0]: tf.constant(inputs, dtype=tf.float32)}
        result     = infer(**kwargs)
        outputs    = [v.numpy() for v in result.values()]
        return outputs if len(outputs) > 1 else outputs[0]
    elif isinstance(model, tf.keras.layers.Layer):
        input_tensor = tf.constant(inputs, dtype=tf.float32)
        result       = model(input_tensor)
        if isinstance(result, (list, tuple)):
            return [r.numpy() if hasattr(r, 'numpy') else np.array(r) for r in result]
        if isinstance(result, dict):
            outputs = [v.numpy() if hasattr(v, 'numpy') else np.array(v) for v in result.values()]
            return outputs if len(outputs) > 1 else outputs[0]
        return result.numpy() if hasattr(result, 'numpy') else np.array(result)
    else:
        input_tensor = tf.constant(inputs, dtype=tf.float32)
        result       = model(input_tensor)
        if isinstance(result, (list, tuple)):
            return [r.numpy() if hasattr(r, 'numpy') else np.array(r) for r in result]
        return result.numpy() if hasattr(result, 'numpy') else np.array(result)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL STATE & LOADING
# ══════════════════════════════════════════════════════════════════════════════

models = {}

CUSTOM_OBJECTS = {
    "SEBlock":               SEBlock,
    "AttentionLayer":        AttentionLayer,
    "FocalLoss":             FocalLoss,
    "L2NormalizationLayer":  L2NormalizationLayer,
    "AnchorBoxLayer":        AnchorBoxLayer,
    "DetectionLoss":         DetectionLoss,
    "WarmupCosineDecay":     WarmupCosineDecay,
}


def load_all_models():
    logger.info("=" * 60)
    logger.info("Loading Spendly AI models...")
    logger.info(f"TensorFlow {tf.__version__}")
    logger.info("=" * 60)

    def _try_load_keras_or_savedmodel(keras_path, savedmodel_path, name, custom_objects=None):
        if os.path.exists(keras_path):
            try:
                m = tf.keras.models.load_model(keras_path, custom_objects=custom_objects)
                return m, ".keras"
            except Exception as e1:
                logger.warning(f"  {name} .keras load failed: {e1}")

        if savedmodel_path and os.path.exists(savedmodel_path):
            try:
                m = tf.keras.layers.TFSMLayer(savedmodel_path, call_endpoint='serving_default')
                return m, "SavedModel (TFSMLayer)"
            except Exception as e2:
                logger.warning(f"  {name} TFSMLayer failed: {e2}")
                try:
                    m = tf.saved_model.load(savedmodel_path)
                    return m, "SavedModel (tf.saved_model)"
                except Exception as e3:
                    logger.warning(f"  {name} tf.saved_model failed: {e3}")

        raise FileNotFoundError(f"No loadable format found for {name}")

    # ── 1. Classifier ─────────────────────────────────────────────────────────
    cls_dir        = os.path.join(MODELS_DIR, "classifier")
    cls_keras_path = os.path.join(cls_dir, "classifier.keras")
    cls_saved_path = os.path.join(cls_dir, "classifier_saved")
    tfidf_path     = os.path.join(cls_dir, "tfidf_vectorizer.joblib")
    nlp_path       = os.path.join(cls_dir, "nlp_preprocessor.joblib")

    if all(os.path.exists(p) for p in [tfidf_path, nlp_path]):
        try:
            cls_model = None
            if os.path.exists(cls_keras_path):
                try:
                    cls_model = tf.keras.models.load_model(
                        cls_keras_path, custom_objects=CUSTOM_OBJECTS
                    )
                    method = ".keras"
                except Exception as e1:
                    logger.warning(f"  Classifier .keras failed: {e1}")

            if cls_model is None and os.path.exists(cls_saved_path):
                sm = tf.saved_model.load(cls_saved_path)
                models["classifier_infer"] = sm.signatures['serving_default']
                cls_model = sm
                method    = "SavedModel (signature)"

            if cls_model is None:
                raise FileNotFoundError("No loadable classifier found")

            tfidf = joblib.load(tfidf_path)
            nlp   = joblib.load(nlp_path)
            models["classifier"] = cls_model
            models["tfidf"]      = tfidf
            models["nlp"]        = nlp
            logger.info(f"  Classifier loaded via {method} (TF-IDF features: {len(tfidf.vocabulary_)})")
        except Exception as e:
            logger.error(f"  Classifier gagal di-load: {e}")
    else:
        missing = [p for p in [tfidf_path, nlp_path] if not os.path.exists(p)]
        logger.warning(f"  Classifier artifacts tidak ditemukan: {missing}")

    # ── 2. Forecaster ─────────────────────────────────────────────────────────
    fc_dir        = os.path.join(MODELS_DIR, "forecaster")
    fc_keras_path = os.path.join(fc_dir, "forecaster.keras")
    fc_saved_path = os.path.join(fc_dir, "forecaster_saved")
    fc_weights    = os.path.join(fc_dir, "forecaster_weights.h5")
    scaler_path   = os.path.join(fc_dir, "scaler.joblib")

    if os.path.exists(scaler_path):
        try:
            try:
                fc_model, method = _try_load_keras_or_savedmodel(
                    fc_keras_path, fc_saved_path, "Forecaster", CUSTOM_OBJECTS
                )
            except FileNotFoundError:
                fc_model = build_forecaster()
                fc_model.load_weights(fc_weights)
                method = ".h5 weights (rebuild)"

            scaler = joblib.load(scaler_path)
            models["forecaster"] = fc_model
            models["scaler"]     = scaler
            logger.info(f"  Forecaster loaded via {method}")
        except Exception as e:
            logger.error(f"  Forecaster gagal di-load: {e}")
    else:
        logger.warning("  Forecaster scaler tidak ditemukan.")

    # ── 3. OCR ────────────────────────────────────────────────────────────────
    ocr_dir        = os.path.join(MODELS_DIR, "ocr")
    ocr_keras_path = os.path.join(ocr_dir, "ocr_fixed.keras")
    ocr_h5_path    = os.path.join(ocr_dir, "ocr_weights_fixed.h5")

    try:
        try:
            ocr_model = tf.keras.models.load_model(
                ocr_keras_path, custom_objects=CUSTOM_OBJECTS
            )
            method = ".keras"
        except Exception:
            if os.path.exists(ocr_h5_path):
                ocr_model = _build_ocr_model()
                ocr_model.load_weights(ocr_h5_path)
                method = ".h5 weights (rebuild)"
            else:
                raise FileNotFoundError(f"No OCR model at {ocr_keras_path} or {ocr_h5_path}")
        models["ocr"] = ocr_model
        logger.info(f"  OCR loaded via {method} (CRNN + CTC)")
    except Exception as e:
        logger.error(f"  OCR gagal di-load: {e}")

    # ── 4. Receipt Detector (MiniYOLO NB07) ───────────────────────────────────
    detector_keras       = os.path.join(MODELS_DIR, "receipt_detector_best.keras")
    detector_keras_final = os.path.join(MODELS_DIR, "receipt_detector_final.keras")
    detector_saved       = os.path.join(MODELS_DIR, "receipt_detector_savedmodel")

    try:
        loaded = False
        for keras_path in [detector_keras, detector_keras_final]:
            if os.path.exists(keras_path):
                try:
                    det_model = tf.keras.models.load_model(
                        keras_path, custom_objects=CUSTOM_OBJECTS
                    )
                    method = f".keras ({os.path.basename(keras_path)})"
                    loaded = True
                    break
                except Exception as e:
                    logger.warning(f"  Detector {keras_path} failed: {e}")

        if not loaded:
            det_model, method = _try_load_keras_or_savedmodel(
                "", detector_saved, "Receipt Detector", CUSTOM_OBJECTS
            )

        models["receipt_detector"] = det_model
        logger.info(f"  Receipt Detector (MiniYOLO 3-class) loaded via {method}")
    except Exception as e:
        logger.error(f"  Receipt Detector gagal di-load: {e}")

    logger.info("-" * 60)
    logger.info(f"Models loaded: {list(models.keys())}")
    logger.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield
    models.clear()
    logger.info("Models unloaded.")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Spendly AI API",
    description=(
        "REST API untuk sistem manajemen keuangan Spendly AI.\n\n"
        "**4 Model ML:**\n"
        "- Receipt Detector: deteksi region date/store/total (MiniYOLO Dual-Scale 3-class)\n"
        "- OCR CRNN: ekstraksi teks per region crop\n"
        "- Multimodal Classifier (TF-IDF + CNN + SE): klasifikasi 9 kategori pengeluaran\n"
        "- LSTM + Attention Forecaster: prediksi pengeluaran mingguan\n\n"
        "**Coding Camp 2026 powered by DBS Foundation — CC26-PSU276**"
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ForecastRequest(BaseModel):
    history: List[List[float]]

    class Config:
        json_schema_extra = {
            "example": {
                "history": [
                    [150000, 500000, 200000, 800000, 100000, 300000, 250000, 350000, 180000]
                ] * 12
            }
        }


class InsightRequest(BaseModel):
    spending_summary: dict

    class Config:
        json_schema_extra = {
            "example": {
                "spending_summary": {
                    "F&B": 500000,
                    "Groceries": 800000,
                    "Beauty": 150000,
                    "Transport": 120000,
                    "total": 1570000,
                }
            }
        }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Status"])
def health_check():
    return {
        "status":   "ok",
        "version":  "2.1.0",
        "categories": CATEGORIES,
        "models_loaded": {
            "receipt_detector": "receipt_detector" in models,
            "ocr":              "ocr" in models,
            "classifier":       "classifier" in models,
            "forecaster":       "forecaster" in models,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /detect-receipt
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/detect-receipt", tags=["Receipt Detector"])
async def detect_receipt_endpoint(file: UploadFile = File(...)):
    """
    Deteksi region date / store / total dalam foto struk menggunakan MiniYOLO Dual-Scale.

    - **Input**: file gambar (JPG/PNG)
    - **Output**: bounding box per region (date, store, total), confidence, dimensi gambar asli
    """
    if "receipt_detector" not in models:
        raise HTTPException(status_code=503, detail="Receipt Detector model belum ter-load.")

    contents = await file.read()
    try:
        img_bgr = bytes_to_bgr(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    orig_h, orig_w = img_bgr.shape[:2]
    detector = models["receipt_detector"]

    try:
        target_h = detector.input_shape[1] or 416
        target_w = detector.input_shape[2] or 416
    except AttributeError:
        target_h, target_w = 416, 416

    img_tensor = preprocess_image_detector(img_bgr, (target_h, target_w))
    det_preds  = model_predict_yolo(detector, img_tensor)
    regions    = decode_detector_output(det_preds, conf_thresh=0.35)

    # Konversi ke pixel coordinates untuk response
    regions_pixel = {}
    for field, det in regions.items():
        cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]
        regions_pixel[field] = {
            "confidence": det["confidence"],
            "bbox_normalized": {
                "cx": round(cx, 4), "cy": round(cy, 4),
                "w":  round(w, 4),  "h":  round(h, 4),
            },
            "bbox_pixels": {
                "x_min": max(0, int((cx - w/2) * orig_w)),
                "y_min": max(0, int((cy - h/2) * orig_h)),
                "x_max": min(orig_w, int((cx + w/2) * orig_w)),
                "y_max": min(orig_h, int((cy + h/2) * orig_h)),
            },
        }

    return {
        "original_size":    {"width": orig_w, "height": orig_h},
        "model_input_size": {"width": target_w, "height": target_h},
        "regions_detected": list(regions.keys()),
        "regions":          regions_pixel,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /ocr
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ocr", tags=["OCR"])
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Ekstraksi teks dari gambar struk menggunakan CRNN + CTC.

    - **Input**: file gambar (JPG/PNG)
    - **Output**: teks hasil OCR
    """
    if "ocr" not in models:
        raise HTTPException(status_code=503, detail="OCR model belum ter-load.")

    contents = await file.read()
    try:
        img_bgr = bytes_to_bgr(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    img_ocr        = preprocess_image_ocr(img_bgr)
    img_tensor     = np.expand_dims(img_ocr, axis=(0, -1))
    logits         = model_predict(models["ocr"], img_tensor)
    extracted_text = decode_ocr(logits)
    blur_info      = assess_blur(img_bgr)

    return {
        "extracted_text": extracted_text,
        "blur_score":     round(blur_info["blur_score"], 2),
        "is_blurry":      blur_info["is_blurry"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /classify
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/classify", tags=["Classifier"])
async def classify_endpoint(
    text: str = Form(default=""),
    file: UploadFile = File(...),
):
    """
    Klasifikasi kategori pengeluaran dari teks struk + gambar.

    - **Input**: `text` (form field) + `file` (gambar struk)
    - **Output**: kategori (9 kelas), confidence score, semua probabilitas
    """
    if "classifier" not in models:
        raise HTTPException(status_code=503, detail="Classifier model belum ter-load.")

    contents = await file.read()
    try:
        img_bgr = bytes_to_bgr(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    blur_info  = assess_blur(img_bgr)
    input_text = "" if blur_info["is_blurry"] else text

    nlp         = models["nlp"]
    tfidf       = models["tfidf"]
    clean_text  = nlp.preprocess(input_text)
    text_vector = tfidf.transform([clean_text]).toarray().astype(np.float32)

    img_arr    = preprocess_image_classifier(img_bgr)
    img_tensor = np.expand_dims(img_arr, axis=0)

    if "classifier_infer" in models:
        infer_fn = models["classifier_infer"]
        result   = infer_fn(
            text_input=tf.constant(text_vector, dtype=tf.float32),
            image_input=tf.constant(img_tensor, dtype=tf.float32),
        )
        preds = next(iter(result.values())).numpy()
    else:
        preds = model_predict(models["classifier"], [text_vector, img_tensor])

    pred_idx   = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])

    return {
        "category":   LABELS[pred_idx],
        "confidence": round(confidence, 4),
        "all_scores": {LABELS[i]: round(float(preds[0][i]), 4) for i in range(NUM_CLASSES)},
        "is_blurry":  blur_info["is_blurry"],
        "blur_score": round(blur_info["blur_score"], 2),
        "text_used":  clean_text if not blur_info["is_blurry"] else "(blurry — CNN only)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /forecast
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/forecast", tags=["Forecaster"])
def forecast_endpoint(request: ForecastRequest):
    """
    Prediksi pengeluaran minggu depan berdasarkan histori 12 minggu.

    - **Input**: `history` — array 12×9 (12 minggu × 9 kategori, dalam Rupiah)
    - **Output**: prediksi pengeluaran minggu ke-13 per kategori
    """
    if "forecaster" not in models:
        raise HTTPException(status_code=503, detail="Forecaster model belum ter-load.")

    history = request.history
    if len(history) != LOOKBACK:
        raise HTTPException(
            status_code=400,
            detail=f"History harus berisi {LOOKBACK} minggu. Diterima: {len(history)}",
        )
    if any(len(row) != N_FORECAST_FEATURES for row in history):
        raise HTTPException(
            status_code=400,
            detail=f"Setiap minggu harus berisi {N_FORECAST_FEATURES} nilai ({CATEGORIES}).",
        )

    scaler      = models["scaler"]
    fc_model    = models["forecaster"]
    history_arr = np.array(history, dtype=np.float32)
    scaled      = scaler.transform(history_arr)
    input_seq   = np.expand_dims(scaled, axis=0)

    pred_scaled = model_predict(fc_model, input_seq)
    pred_real   = scaler.inverse_transform(pred_scaled)[0]

    return {
        "prediction": {cat: round(float(pred_real[i])) for i, cat in enumerate(CATEGORIES)},
        "unit":       "IDR (Rupiah)",
        "horizon":    "1 minggu ke depan",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /insight
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/insight", tags=["Generative AI"])
async def insight_endpoint(request: InsightRequest):
    """
    Saran keuangan personal menggunakan Google Gemini Generative AI.

    - **Input**: `spending_summary` — dict ringkasan pengeluaran per kategori
    - **Output**: saran keuangan berbahasa Indonesia
    """
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY tidak ditemukan.")

    summary     = request.spending_summary
    summary_str = "\n".join(
        f"  - {k}: Rp {v:,.0f}" if isinstance(v, (int, float)) else f"  - {k}: {v}"
        for k, v in summary.items()
    )

    prompt = (
        f"Kamu adalah asisten keuangan pribadi untuk generasi muda Indonesia.\n\n"
        f"Berikut ringkasan pengeluaran pengguna minggu ini:\n{summary_str}\n\n"
        f"Berikan 3 saran keuangan yang praktis, spesifik, dan mudah dipahami "
        f"berdasarkan data di atas. Gunakan bahasa Indonesia yang ramah dan motivatif. "
        f"Format: poin-poin singkat."
    )

    gemini_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            gemini_url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 512, "temperature": 0.7},
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {response.text}")

    result    = response.json()
    ai_advice = result["candidates"][0]["content"]["parts"][0]["text"]

    return {
        "spending_summary": summary,
        "insight":          ai_advice,
        "model_used":       "gemini-2.5-flash",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /process-receipt  (Full Pipeline)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/process-receipt", tags=["Full Pipeline"])
async def process_receipt_endpoint(file: UploadFile = File(...)):
    """
    Full pipeline: Upload foto → Deteksi region (date/store/total) → OCR per region → Klasifikasi.

    1. Receipt region detection (MiniYOLO Dual-Scale 3-class)
    2. OCR per region (CRNN + CTC) → parsed_fields: date, store, total
    3. Blur detection pada gambar penuh
    4. OCR gambar penuh → extracted_text untuk classifier
    5. Multimodal classification (teks + gambar)

    - **Input**: file gambar struk (JPG/PNG)
    - **Output**: parsed_fields (date/store/total), extracted_text, kategori, confidence
    """
    for required in ["ocr", "classifier"]:
        if required not in models:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{required}' belum ter-load.",
            )

    contents = await file.read()
    try:
        img_bgr = bytes_to_bgr(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    orig_h, orig_w = img_bgr.shape[:2]

    # ── Step 1: Detect region date/store/total ────────────────────────────
    parsed_fields  = {"store": None, "date": None, "total": None}
    detection_info = {"performed": False, "regions_detected": [], "region_confidences": {}}

    if "receipt_detector" in models:
        detection_info["performed"] = True
        try:
            detector = models["receipt_detector"]
            try:
                target_h = detector.input_shape[1] or 416
                target_w = detector.input_shape[2] or 416
            except AttributeError:
                target_h, target_w = 416, 416

            det_tensor = preprocess_image_detector(img_bgr, (target_h, target_w))
            det_preds  = model_predict_yolo(detector, det_tensor)
            regions    = decode_detector_output(det_preds, conf_thresh=0.35)

            detection_info["regions_detected"]    = list(regions.keys())
            detection_info["region_confidences"]  = {
                k: v["confidence"] for k, v in regions.items()
            }

            # ── Step 2: OCR per region ────────────────────────────────────
            for field in ["store", "date", "total"]:
                if field in regions:
                    crop = crop_region(img_bgr, regions[field], padding=0.01)
                    text = ocr_region(crop)
                    parsed_fields[field] = text.strip() if text.strip() else None
                    logger.info(f"  OCR [{field}]: '{parsed_fields[field]}'")

        except Exception as e:
            logger.warning(f"Detection/OCR region gagal: {e}")
            detection_info["error"] = str(e)

    # ── Step 3: Blur detection ────────────────────────────────────────────
    blur_info   = assess_blur(img_bgr)
    is_blurry   = blur_info["is_blurry"]
    ocr_skipped = is_blurry

    # ── Step 4: OCR gambar penuh (untuk classifier) ───────────────────────
    extracted_text = ""
    if not is_blurry:
        img_ocr        = preprocess_image_ocr(img_bgr)
        img_tensor     = np.expand_dims(img_ocr, axis=(0, -1))
        logits         = model_predict(models["ocr"], img_tensor)
        extracted_text = decode_ocr(logits)

    # ── Step 5: Multimodal classification ────────────────────────────────
    # Gabungkan semua teks hasil OCR region sebagai input classifier
    combined_text = " ".join(filter(None, [
        parsed_fields.get("store"),
        parsed_fields.get("date"),
        parsed_fields.get("total"),
        extracted_text,
    ]))
    input_text  = "" if is_blurry else combined_text

    nlp         = models["nlp"]
    tfidf       = models["tfidf"]
    clean_text  = nlp.preprocess(input_text)
    text_vector = tfidf.transform([clean_text]).toarray().astype(np.float32)

    img_arr    = preprocess_image_classifier(img_bgr)
    img_tensor = np.expand_dims(img_arr, axis=0)

    if "classifier_infer" in models:
        infer_fn = models["classifier_infer"]
        result   = infer_fn(
            text_input=tf.constant(text_vector, dtype=tf.float32),
            image_input=tf.constant(img_tensor, dtype=tf.float32),
        )
        preds = next(iter(result.values())).numpy()
    else:
        preds = model_predict(models["classifier"], [text_vector, img_tensor])

    pred_idx   = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])

    return {
        "parsed_fields":  parsed_fields,
        "extracted_text": extracted_text,
        "ocr_skipped":    ocr_skipped,
        "category":       LABELS[pred_idx],
        "confidence":     round(confidence, 4),
        "all_scores":     {LABELS[i]: round(float(preds[0][i]), 4) for i in range(NUM_CLASSES)},
        "is_blurry":      is_blurry,
        "blur_score":     round(blur_info["blur_score"], 2),
        "detection_info": detection_info,
    }