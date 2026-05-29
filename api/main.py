"""
Spendly AI — FastAPI Inference Server v2.0
===========================================
REST API untuk serving 4 model ML Spendly:
  - Receipt Detector : deteksi area struk dalam foto (CNN)
  - OCR CRNN         : ekstraksi teks dari gambar struk
  - Classifier       : klasifikasi 9 kategori pengeluaran (TF-IDF + CNN + SE)
  - Forecaster       : prediksi pengeluaran mingguan (LSTM + Attention)

Endpoint tambahan:
  - /insight          : saran keuangan via Google Gemini Generative AI
  - /process-receipt  : full pipeline (Detect → OCR → Classify)
  - /detect-receipt   : deteksi bounding box struk dalam foto
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

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spendly-api")

# ── Path Setup ────────────────────────────────────────────────────────────────
API_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_DIR)
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

# Add src/ to sys.path so SavedModel can find custom_components module
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ── Konstanta Global ──────────────────────────────────────────────────────────
# 9 kategori — Transport baru ditambahkan (belum ada data histori)
CATEGORIES = [
    "Beauty", "F&B", "Gas", "Groceries", "Health",
    "HouseHold", "Lifestyle", "Listrik", "Transport",
]
LABELS       = CATEGORIES
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
NUM_CLASSES  = len(LABELS)            # 9
IMG_SIZE     = 224
LOOKBACK     = 12

## Forecaster juga dilatih pada 9 kategori (NB04 v2 — Transport ditambahkan)
# scaler.n_features_in_ = 9, Dense(9)
N_FORECAST_FEATURES = NUM_CLASSES  # 9

BLUR_THRESHOLD = 100

# OCR Charset — identik dengan NB05
CHARS_OCR     = string.digits + string.ascii_uppercase + string.ascii_lowercase + " .,:-/()%"
BLANK_IDX     = len(CHARS_OCR) + 1    # 72
NUM_OCR_CHARS = len(CHARS_OCR) + 2    # 73
OCR_IMG_W     = 400
OCR_IMG_H     = 50


# ══════════════════════════════════════════════════════════════════════════════
# INLINE CUSTOM KERAS COMPONENTS (self-contained — no src/ imports)
# ══════════════════════════════════════════════════════════════════════════════

# ── Sastrawi (optional) ──────────────────────────────────────────────────────
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    logger.warning("PySastrawi tidak tersedia. NLP preprocessing akan di-skip.")


class NLPPreprocessor:
    """Pipeline NLP untuk teks transaksi Bahasa Indonesia.
    Identik dengan definisi di NB03 & NB06 dan custom_components.py.
    """
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


# ── SEBlock (Squeeze-and-Excitation) — identik NB06 / custom_components.py ──
class SEBlock(tf.keras.layers.Layer):
    """Squeeze-and-Excitation Block untuk CNN branch classifier."""

    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap    = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(
            max(1, channels // self.ratio), activation="relu", use_bias=False
        )
        self.dense2 = tf.keras.layers.Dense(
            channels, activation="sigmoid", use_bias=False
        )
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


# ── AttentionLayer — identik NB04 / custom_components.py ────────────────────
class AttentionLayer(tf.keras.layers.Layer):
    """Custom Attention Layer untuk sequence modeling (LSTM forecaster).

    Input shape : (batch, timesteps, features)
    Output shape: (batch, features)
    """
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W(inputs)))                  # (batch, T, 1)
        attention_weights = tf.nn.softmax(score, axis=1)            # (batch, T, 1)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ── FocalLoss — identik custom_components.py (needed for .keras loading) ────
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss dengan per-class alpha untuk class imbalance."""

    def __init__(self, gamma=2.0, alpha=0.25, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights

        if class_weights is not None:
            vals = list(class_weights.values())
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
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce     = -y_true * tf.math.log(y_pred)
        pt     = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal  = tf.math.pow(1.0 - pt, self.gamma) * ce

        if isinstance(self._alpha_tensor, float):
            weighted = self._alpha_tensor * focal
        else:
            alpha_t  = tf.reshape(self._alpha_tensor, (1, -1))
            weighted = alpha_t * y_true * focal

        return tf.reduce_mean(tf.reduce_sum(weighted, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "class_weights": self.class_weights,
        })
        return config


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDER — hanya forecaster yang perlu di-rebuild (menggunakan .h5)
# ══════════════════════════════════════════════════════════════════════════════

def build_forecaster(lookback: int = LOOKBACK, n_features: int = N_FORECAST_FEATURES):
    """Bangun arsitektur forecaster — identik dengan NB04 v2.

    LSTM(64, ret_seq) → LSTM(32, ret_seq) → AttentionLayer → Dense(32) → Dense(9)
    """
    inputs = tf.keras.Input(shape=(lookback, n_features))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x = AttentionLayer(name="attention")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_features)(x)
    return tf.keras.Model(inputs, outputs, name="spendly_forecaster_attention")


def _build_ocr_model():
    """Rebuild OCR CRNN architecture — identik dengan NB05.

    3×Conv+BN+MaxPool → SpatialDropout → Reshape → 2×BiLSTM → Dense → Dense(73)
    Hanya digunakan jika .keras file tidak kompatibel dengan Keras versi ini.
    """
    from tensorflow.keras import layers

    img_w, img_h = 400, 50
    num_chars = NUM_OCR_CHARS  # 73

    inputs = tf.keras.Input(shape=(img_h, img_w, 1), name="ocr_input")

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)       # → (25, 200, 32)

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)       # → (12, 100, 64)

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)       # → (6, 100, 128)

    # Conv Block 4 + SpatialDropout
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.3)(x)                 # → (6, 100, 128)

    # Reshape for RNN: (batch, time=100, features=768)
    x = layers.Permute((2, 1, 3))(x)                    # → (100, 6, 128)
    x = layers.Reshape((-1, 6 * 128))(x)                # → (100, 768)

    # BiLSTM layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Dense output
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_chars, activation="linear", name="logits")(x)  # → (100, 73)

    return tf.keras.Model(inputs, outputs, name="spendly_ocr_crnn")


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def assess_blur(image_bgr: np.ndarray) -> dict:
    """Laplacian variance blur detection — identik NB01."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {"blur_score": blur_score, "is_blurry": blur_score < BLUR_THRESHOLD}


def preprocess_image_classifier(image_bgr: np.ndarray) -> np.ndarray:
    """Load & resize untuk classifier — identik NB03."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype(np.float32) / 255.0


def preprocess_image_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE + Adaptive Threshold + Denoise — identik NB05."""
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
    gray = cv2.resize(gray, (OCR_IMG_W, OCR_IMG_H))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    return gray.astype(np.float32) / 255.0


def preprocess_image_detector(image_bgr: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize dan normalize gambar untuk receipt detector.

    Args:
        image_bgr: input BGR image
        target_size: (height, width) sesuai input shape model
    Returns:
        normalized float32 array (1, H, W, 3)
    """
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[1], target_size[0]))  # cv2 uses (W, H)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def decode_ocr(logits: np.ndarray) -> str:
    """CTC greedy decode — identik NB05."""
    logits = tf.cast(logits, tf.float32)
    pred_indices = tf.argmax(logits, axis=-1).numpy()[0]
    text = ""
    prev_idx = -1
    for idx in pred_indices:
        if idx != prev_idx:
            if 1 <= idx <= len(CHARS_OCR):
                text += CHARS_OCR[idx - 1]
        prev_idx = idx
    return text


def bytes_to_bgr(file_bytes: bytes) -> np.ndarray:
    """Konversi bytes upload ke BGR numpy array."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak dapat dibaca. Pastikan format JPG/PNG.")
    return img


def model_predict(model, inputs):
    """Universal prediction: works with Keras Model, TFSMLayer, or saved_model object.

    - Keras Model: uses model.predict()
    - TFSMLayer: calls model(inputs) — returns dict, extract first output
    - tf.saved_model: calls model.signatures['serving_default'](input_tensor)

    `inputs` can be a single array or a list of arrays (multi-input model).
    """
    if hasattr(model, 'predict'):
        # Standard Keras Model
        return model.predict(inputs, verbose=0)
    elif isinstance(model, tf.keras.layers.Layer):
        # TFSMLayer — needs tensor inputs as keyword args
        if isinstance(inputs, (list, tuple)):
            # Multi-input model: convert each to tensor
            tensor_inputs = [
                tf.constant(x, dtype=tf.float32) if not isinstance(x, tf.Tensor) else x
                for x in inputs
            ]
            # TFSMLayer wraps a SavedModel — we need to call it with named kwargs
            # matching the input tensor names. Use tf.saved_model.load() to find them.
            endpoint_fn = None
            input_keys = None
            # Try to get input names from the layer's endpoint
            for attr in ('_callable', '_endpoint'):
                fn = getattr(model, attr, None)
                if fn is not None and hasattr(fn, 'structured_input_signature'):
                    input_keys = list(fn.structured_input_signature[1].keys())
                    break
            if input_keys is None:
                # Reload SavedModel to discover input names
                saved_path = getattr(model, '_asset_path', None) or getattr(model, 'filepath', None)
                if saved_path:
                    sm = tf.saved_model.load(str(saved_path))
                    sig = sm.signatures.get('serving_default')
                    if sig:
                        input_keys = list(sig.structured_input_signature[1].keys())
                        endpoint_fn = sig
            if input_keys and len(input_keys) >= len(tensor_inputs):
                kwargs = {input_keys[i]: tensor_inputs[i] for i in range(len(tensor_inputs))}
                if endpoint_fn is not None:
                    result = endpoint_fn(**kwargs)
                else:
                    result = model(**kwargs)
            else:
                # Last fallback: call with dict positional
                result = model(tensor_inputs[0], tensor_inputs[1] if len(tensor_inputs) > 1 else None)
        else:
            input_tensor = tf.constant(inputs, dtype=tf.float32) if not isinstance(inputs, tf.Tensor) else inputs
            result = model(input_tensor)

        if isinstance(result, dict):
            val = next(iter(result.values()))
            return val.numpy() if hasattr(val, 'numpy') else np.array(val)
        return result.numpy() if hasattr(result, 'numpy') else np.array(result)
    elif hasattr(model, 'signatures'):
        # tf.saved_model object
        infer = model.signatures['serving_default']
        input_keys = list(infer.structured_input_signature[1].keys())
        if isinstance(inputs, (list, tuple)):
            kwargs = {input_keys[i]: tf.constant(inputs[i], dtype=tf.float32) for i in range(min(len(input_keys), len(inputs)))}
        else:
            kwargs = {input_keys[0]: tf.constant(inputs, dtype=tf.float32)}
        result = infer(**kwargs)
        val = next(iter(result.values()))
        return val.numpy()
    else:
        # Fallback
        if isinstance(inputs, (list, tuple)):
            tensors = [tf.constant(x, dtype=tf.float32) for x in inputs]
            result = model(*tensors)
        else:
            result = model(tf.constant(inputs, dtype=tf.float32))
        return result.numpy() if hasattr(result, 'numpy') else np.array(result)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL STATE & LOADING
# ══════════════════════════════════════════════════════════════════════════════
models = {}

# Custom objects dict untuk tf.keras.models.load_model()
CUSTOM_OBJECTS = {
    "SEBlock": SEBlock,
    "AttentionLayer": AttentionLayer,
    "FocalLoss": FocalLoss,
}


def load_all_models():
    """Load semua model saat startup.

    Strategi loading (kompatibel TF 2.10 dan TF 2.21/Keras 3):
      1. Coba tf.keras.models.load_model() pada .keras file
      2. Fallback ke tf.saved_model.load() pada SavedModel dir
      3. Fallback ke rebuild architecture + load .h5 weights
    """
    logger.info("=" * 60)
    logger.info("Loading Spendly AI models...")
    logger.info(f"TensorFlow {tf.__version__}")
    logger.info("=" * 60)

    def _try_load_keras_or_savedmodel(keras_path, savedmodel_path, name, custom_objects=None):
        """Try .keras first, then SavedModel, return (model, method_str) or raise."""
        # 1. Try .keras (works on TF 2.10-style .keras = HDF5, or Keras 3 ZIP)
        if os.path.exists(keras_path):
            try:
                m = tf.keras.models.load_model(keras_path, custom_objects=custom_objects)
                return m, ".keras"
            except Exception as e1:
                logger.warning(f"  {name} .keras load failed: {e1}")

        # 2. Try SavedModel directory
        if savedmodel_path and os.path.exists(savedmodel_path):
            try:
                # Keras 3 approach: TFSMLayer wraps SavedModel as inference layer
                m = tf.keras.layers.TFSMLayer(savedmodel_path, call_endpoint='serving_default')
                return m, "SavedModel (TFSMLayer)"
            except Exception as e2:
                logger.warning(f"  {name} TFSMLayer failed: {e2}")
                try:
                    # Low-level tf.saved_model.load
                    m = tf.saved_model.load(savedmodel_path)
                    return m, "SavedModel (tf.saved_model)"
                except Exception as e3:
                    logger.warning(f"  {name} tf.saved_model failed: {e3}")

        raise FileNotFoundError(f"No loadable format found for {name}")

    # ── 1. Classifier ────────────────────────────────────────────────────────
    cls_dir        = os.path.join(MODELS_DIR, "classifier")
    cls_keras_path = os.path.join(cls_dir, "classifier.keras")
    cls_saved_path = os.path.join(cls_dir, "classifier_saved")
    tfidf_path     = os.path.join(cls_dir, "tfidf_vectorizer.joblib")
    nlp_path       = os.path.join(cls_dir, "nlp_preprocessor.joblib")

    if all(os.path.exists(p) for p in [tfidf_path, nlp_path]):
        try:
            # Classifier has multi-input (text + image) — use tf.saved_model.load()
            # instead of TFSMLayer which has issues with multi-input kwargs
            if os.path.exists(cls_keras_path):
                try:
                    cls_model = tf.keras.models.load_model(
                        cls_keras_path, custom_objects=CUSTOM_OBJECTS
                    )
                    method = ".keras"
                except Exception as e1:
                    logger.warning(f"  Classifier .keras failed: {e1}")
                    cls_model = None
            else:
                cls_model = None

            if cls_model is None and os.path.exists(cls_saved_path):
                sm = tf.saved_model.load(cls_saved_path)
                # Store the signature function — it takes text_input + image_input kwargs
                models["classifier_infer"] = sm.signatures['serving_default']
                cls_model = sm  # Store raw saved_model
                method = "SavedModel (signature)"

            if cls_model is None:
                raise FileNotFoundError("No loadable classifier found")

            tfidf = joblib.load(tfidf_path)
            nlp   = joblib.load(nlp_path)
            models["classifier"] = cls_model
            models["tfidf"]      = tfidf
            models["nlp"]        = nlp
            n_features = len(tfidf.vocabulary_)
            logger.info(f"  Classifier loaded via {method} (TF-IDF features: {n_features})")
        except Exception as e:
            logger.error(f"  Classifier gagal di-load: {e}")
    else:
        missing = [p for p in [tfidf_path, nlp_path] if not os.path.exists(p)]
        logger.warning(f"  Classifier artifacts tidak ditemukan: {missing}")

    # ── 2. Forecaster ────────────────────────────────────────────────────────
    fc_dir         = os.path.join(MODELS_DIR, "forecaster")
    fc_keras_path  = os.path.join(fc_dir, "forecaster.keras")
    fc_saved_path  = os.path.join(fc_dir, "forecaster_saved")
    fc_weights     = os.path.join(fc_dir, "forecaster_weights.h5")
    scaler_path    = os.path.join(fc_dir, "scaler.joblib")

    if os.path.exists(scaler_path):
        try:
            try:
                fc_model, method = _try_load_keras_or_savedmodel(
                    fc_keras_path, fc_saved_path, "Forecaster", CUSTOM_OBJECTS
                )
            except FileNotFoundError:
                # Last resort: rebuild architecture + load weights
                fc_model = build_forecaster()
                fc_model.load_weights(fc_weights)
                method = ".h5 weights (rebuild)"

            scaler = joblib.load(scaler_path)
            models["forecaster"] = fc_model
            models["scaler"]     = scaler
            logger.info(f"  Forecaster loaded via {method} (scaler n_features={scaler.n_features_in_})")
        except Exception as e:
            logger.error(f"  Forecaster gagal di-load: {e}")
    else:
        logger.warning("  Forecaster scaler tidak ditemukan.")

    # ── 3. OCR ───────────────────────────────────────────────────────────────
    ocr_dir        = os.path.join(MODELS_DIR, "ocr")
    ocr_keras_path = os.path.join(ocr_dir, "ocr_fixed.keras")
    ocr_h5_path    = os.path.join(ocr_dir, "ocr_weights_fixed.h5")

    try:
        # Try .keras first
        try:
            ocr_model = tf.keras.models.load_model(
                ocr_keras_path, custom_objects=CUSTOM_OBJECTS
            )
            method = ".keras"
        except Exception:
            # Rebuild CRNN architecture and load .h5 weights
            if os.path.exists(ocr_h5_path):
                ocr_model = _build_ocr_model()
                ocr_model.load_weights(ocr_h5_path)
                method = ".h5 weights (rebuild)"
            else:
                raise FileNotFoundError(f"No OCR model found at {ocr_keras_path} or {ocr_h5_path}")
        models["ocr"] = ocr_model
        logger.info(f"  OCR loaded via {method} (CRNN + CTC)")
    except Exception as e:
        logger.error(f"  OCR gagal di-load: {e}")

    # ── 4. Receipt Detector ──────────────────────────────────────────────────
    detector_keras = os.path.join(MODELS_DIR, "receipt_detector_best.keras")
    detector_saved = os.path.join(MODELS_DIR, "receipt_detector_savedmodel")

    try:
        det_model, method = _try_load_keras_or_savedmodel(
            detector_keras, detector_saved, "Receipt Detector", CUSTOM_OBJECTS
        )
        models["receipt_detector"] = det_model
        logger.info(f"  Receipt Detector loaded via {method}")
    except Exception as e:
        logger.error(f"  Receipt Detector gagal di-load: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info(f"Models loaded: {list(models.keys())}")
    logger.info(f"Categories: {CATEGORIES} ({NUM_CLASSES} total)")
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
        "- Receipt Detector: deteksi area struk dalam foto\n"
        "- OCR CRNN: ekstraksi teks dari foto struk\n"
        "- Multimodal Classifier (TF-IDF + CNN + SE): klasifikasi 9 kategori pengeluaran\n"
        "- LSTM + Attention Forecaster: prediksi pengeluaran mingguan\n\n"
        "**Fitur Tambahan:**\n"
        "- Gemini AI Insight: saran keuangan personal\n\n"
        "**Coding Camp 2026 powered by DBS Foundation — CC26-PSU276**"
    ),
    version="2.0.0",
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

class ClassifyRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "INDOMARET Total Rp 45.000 Sabun Mandi Pasta Gigi"}
        }


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
    """Cek status server dan model yang ter-load."""
    return {
        "status": "ok",
        "version": "2.0.0",
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
    Deteksi area struk (bounding box) dalam foto menggunakan CNN.

    - **Input**: file gambar (JPG/PNG)
    - **Output**: bounding boxes, confidence scores, dan dimensi gambar asli
    """
    if "receipt_detector" not in models:
        raise HTTPException(
            status_code=503,
            detail="Receipt Detector model belum ter-load.",
        )

    contents = await file.read()
    try:
        img_bgr = bytes_to_bgr(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    detector = models["receipt_detector"]
    orig_h, orig_w = img_bgr.shape[:2]

    # Ambil input shape dari model secara dinamis
    model_input_shape = detector.input_shape  # (None, H, W, C)
    target_h = model_input_shape[1] if model_input_shape[1] else 224
    target_w = model_input_shape[2] if model_input_shape[2] else 224

    img_tensor = preprocess_image_detector(img_bgr, (target_h, target_w))
    predictions = model_predict(detector, img_tensor)

    # Format output tergantung arsitektur model
    # Asumsi output: bounding box coordinates (normalized) dan/atau confidence
    result = {
        "original_size": {"width": orig_w, "height": orig_h},
        "model_input_size": {"width": target_w, "height": target_h},
    }

    if isinstance(predictions, list):
        # Multi-output model
        result["raw_predictions"] = [p.tolist() for p in predictions]
    else:
        pred_array = predictions[0]  # batch dim
        if pred_array.ndim == 1:
            if len(pred_array) >= 4:
                # Output = [x_min, y_min, x_max, y_max, ...confidence]
                bbox = pred_array[:4].tolist()
                confidence = float(pred_array[4]) if len(pred_array) > 4 else 1.0
                # Denormalize bbox ke koordinat piksel asli
                result["detections"] = [
                    {
                        "bbox_normalized": {
                            "x_min": round(bbox[0], 4),
                            "y_min": round(bbox[1], 4),
                            "x_max": round(bbox[2], 4),
                            "y_max": round(bbox[3], 4),
                        },
                        "bbox_pixels": {
                            "x_min": int(bbox[0] * orig_w),
                            "y_min": int(bbox[1] * orig_h),
                            "x_max": int(bbox[2] * orig_w),
                            "y_max": int(bbox[3] * orig_h),
                        },
                        "confidence": round(confidence, 4),
                    }
                ]
            else:
                # Binary classification (receipt vs no receipt)
                confidence = float(pred_array[0]) if len(pred_array) == 1 else float(np.max(pred_array))
                result["is_receipt"] = bool(confidence > 0.5)
                result["confidence"] = round(confidence, 4)
        else:
            # Multiple detections or grid output
            result["raw_predictions"] = pred_array.tolist()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /ocr
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ocr", tags=["OCR"])
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Ekstraksi teks dari gambar struk belanja menggunakan CRNN + CTC.

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

    img_ocr = preprocess_image_ocr(img_bgr)
    img_tensor = np.expand_dims(img_ocr, axis=(0, -1))  # (1, H, W, 1)

    logits = model_predict(models["ocr"], img_tensor)
    extracted_text = decode_ocr(logits)

    blur_info = assess_blur(img_bgr)

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
    text: str = Form(default="", description="Teks OCR dari struk (bisa kosong jika gambar blurry)"),
    file: UploadFile = File(...),
):
    """
    Klasifikasi kategori pengeluaran dari teks struk + gambar.

    - **Input**: `text` (form field, teks OCR) + `file` (gambar struk)
    - **Output**: kategori (9 kelas), confidence score, semua probabilitas

    Gunakan `multipart/form-data`. Contoh curl:
    ```
    curl -X POST /classify -F "text=INDOMARET Total Rp 45000" -F "file=@struk.jpg"
    ```
    """
    if "classifier" not in models:
        raise HTTPException(status_code=503, detail="Classifier model belum ter-load.")

    contents = await file.read()
    try:
        img_bgr = bytes_to_bgr(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Blur check → zero text jika blurry
    blur_info  = assess_blur(img_bgr)
    input_text = "" if blur_info["is_blurry"] else text

    # NLP Preprocessing → TF-IDF
    nlp   = models["nlp"]
    tfidf = models["tfidf"]
    clean_text  = nlp.preprocess(input_text)
    text_vector = tfidf.transform([clean_text]).toarray().astype(np.float32)

    # Image preprocessing
    img_arr    = preprocess_image_classifier(img_bgr)
    img_tensor = np.expand_dims(img_arr, axis=0)  # (1, 224, 224, 3)

    # Use classifier_infer (SavedModel signature) if available, else model_predict
    if "classifier_infer" in models:
        infer_fn = models["classifier_infer"]
        result = infer_fn(
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
        "all_scores": {
            LABELS[i]: round(float(preds[0][i]), 4) for i in range(NUM_CLASSES)
        },
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
    - **Output**: prediksi pengeluaran minggu ke-13 per kategori (9 kategori)

    Urutan 9 kategori input: Beauty, F&B, Gas, Groceries, Health, HouseHold, Lifestyle, Listrik, Transport.
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
            detail=(
                f"Setiap minggu harus berisi {N_FORECAST_FEATURES} nilai "
                f"(satu per kategori: {CATEGORIES})."
            ),
        )

    scaler   = models["scaler"]
    fc_model = models["forecaster"]

    history_arr = np.array(history, dtype=np.float32)       # (12, 9)
    scaled      = scaler.transform(history_arr)              # normalize
    input_seq   = np.expand_dims(scaled, axis=0)             # (1, 12, 9)

    pred_scaled = model_predict(fc_model, input_seq)     # (1, 9)
    pred_real   = scaler.inverse_transform(pred_scaled)[0]   # (9,) Rupiah

    prediction = {}
    for i, cat in enumerate(CATEGORIES):
        prediction[cat] = round(float(pred_real[i]))

    return {
        "prediction": prediction,
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

    *Membutuhkan GEMINI_API_KEY di environment variable.*
    """
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY tidak ditemukan di environment.",
        )

    summary = request.spending_summary
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
                "contents": [
                    {"parts": [{"text": prompt}]}
                ],
                "generationConfig": {
                    "maxOutputTokens": 512,
                    "temperature": 0.7,
                },
            },
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {response.text}",
        )

    result    = response.json()
    ai_advice = result["candidates"][0]["content"]["parts"][0]["text"]

    return {
        "spending_summary": summary,
        "insight":          ai_advice,
        "model_used":       "gemini-2.5-flash",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: /process-receipt  (Full Pipeline: Detect → OCR → Classify)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/process-receipt", tags=["Full Pipeline"])
async def process_receipt_endpoint(file: UploadFile = File(...)):
    """
    Full pipeline: Upload foto → Deteksi Struk → OCR → Klasifikasi Kategori.

    1. Receipt detection (CNN) — crop area struk jika terdeteksi
    2. Blur detection (Laplacian variance)
    3. OCR (CRNN + CTC) — di-skip jika gambar blurry
    4. Multimodal classification (teks OCR + gambar)

    - **Input**: file gambar struk (JPG/PNG)
    - **Output**: teks OCR, kategori, confidence, info blur, info deteksi
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
    detection_info = {"performed": False, "receipt_cropped": False}

    # ── Step 1: Receipt Detection (optional — jika model tersedia) ───────
    img_for_processing = img_bgr
    if "receipt_detector" in models:
        detection_info["performed"] = True
        try:
            detector = models["receipt_detector"]
            model_input_shape = detector.input_shape
            target_h = model_input_shape[1] if model_input_shape[1] else 224
            target_w = model_input_shape[2] if model_input_shape[2] else 224

            det_tensor = preprocess_image_detector(img_bgr, (target_h, target_w))
            det_preds  = model_predict(detector, det_tensor)

            # Coba crop bounding box jika output berupa koordinat
            if not isinstance(det_preds, list):
                pred_array = det_preds[0]
                if pred_array.ndim == 1 and len(pred_array) >= 4:
                    x_min = max(0, int(pred_array[0] * orig_w))
                    y_min = max(0, int(pred_array[1] * orig_h))
                    x_max = min(orig_w, int(pred_array[2] * orig_w))
                    y_max = min(orig_h, int(pred_array[3] * orig_h))

                    # Hanya crop jika bbox cukup besar (minimal 10% area asli)
                    bbox_area  = (x_max - x_min) * (y_max - y_min)
                    orig_area  = orig_w * orig_h
                    if bbox_area > 0.10 * orig_area and x_max > x_min and y_max > y_min:
                        img_for_processing = img_bgr[y_min:y_max, x_min:x_max]
                        detection_info["receipt_cropped"] = True
                        detection_info["bbox_pixels"] = {
                            "x_min": x_min, "y_min": y_min,
                            "x_max": x_max, "y_max": y_max,
                        }
                        logger.info(
                            f"Receipt cropped: ({x_min},{y_min}) → ({x_max},{y_max})"
                        )
        except Exception as e:
            logger.warning(f"Receipt detection gagal, pakai gambar asli: {e}")
            detection_info["error"] = str(e)

    # ── Step 2: Blur detection ───────────────────────────────────────────
    blur_info   = assess_blur(img_for_processing)
    is_blurry   = blur_info["is_blurry"]
    ocr_skipped = is_blurry

    # ── Step 3: OCR (skip jika blurry) ───────────────────────────────────
    extracted_text = ""
    if not is_blurry:
        img_ocr    = preprocess_image_ocr(img_for_processing)
        img_tensor = np.expand_dims(img_ocr, axis=(0, -1))
        logits     = model_predict(models["ocr"], img_tensor)
        extracted_text = decode_ocr(logits)

    # ── Step 4: Multimodal classification ────────────────────────────────
    input_text  = "" if is_blurry else extracted_text
    nlp         = models["nlp"]
    tfidf       = models["tfidf"]
    clean_text  = nlp.preprocess(input_text)
    text_vector = tfidf.transform([clean_text]).toarray().astype(np.float32)

    img_arr    = preprocess_image_classifier(img_for_processing)
    img_tensor = np.expand_dims(img_arr, axis=0)

    if "classifier_infer" in models:
        infer_fn = models["classifier_infer"]
        result = infer_fn(
            text_input=tf.constant(text_vector, dtype=tf.float32),
            image_input=tf.constant(img_tensor, dtype=tf.float32),
        )
        preds = next(iter(result.values())).numpy()
    else:
        preds = model_predict(models["classifier"], [text_vector, img_tensor])
    pred_idx   = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])

    return {
        "extracted_text":  extracted_text,
        "ocr_skipped":     ocr_skipped,
        "category":        LABELS[pred_idx],
        "confidence":      round(confidence, 4),
        "all_scores":      {
            LABELS[i]: round(float(preds[0][i]), 4) for i in range(NUM_CLASSES)
        },
        "is_blurry":       is_blurry,
        "blur_score":      round(blur_info["blur_score"], 2),
        "detection_info":  detection_info,
    }
