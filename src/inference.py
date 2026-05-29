"""
Spendly AI — Inference Module (v2 — Synced with A100 notebooks)
================================================================
Coding Camp 2026 powered by DBS Foundation (CC26-PSU276)

Updated: Synced OCR architecture with NB05 v6, classifier with NB03 v4,
         forecaster with NB04, and unified preprocessing.
"""
import os, re, string
import numpy as np
import tensorflow as tf
import joblib
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import shared constants from custom_components
try:
    from custom_components import (
        CATEGORIES, NUM_CATEGORIES, CLASSIFIER_IMG_SIZE,
        BLUR_THRESHOLD, OCR_IMG_H, OCR_IMG_W,
        preprocess_for_ocr, decode_ocr_prediction, OCR_CHARS,
        SEBlock, AttentionLayer, FocalLoss
    )
    LABELS = CATEGORIES
except ImportError:
    LABELS = ["Beauty", "F&B", "Gas", "Groceries", "Health",
              "HouseHold", "Lifestyle", "Listrik", "Transport"]

    # ── Inline fallback: SEBlock ──
    class SEBlock(tf.keras.layers.Layer):
        def __init__(self, ratio=8, **kwargs):
            super().__init__(**kwargs)
            self.ratio = ratio

        def build(self, input_shape):
            channels = input_shape[-1]
            self.gap = tf.keras.layers.GlobalAveragePooling2D()
            self.dense1 = tf.keras.layers.Dense(
                max(1, channels // self.ratio), activation='relu', use_bias=False
            )
            self.dense2 = tf.keras.layers.Dense(channels, activation='sigmoid', use_bias=False)
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

    # ── Inline fallback: AttentionLayer ──
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, units=64, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.W = tf.keras.layers.Dense(units, use_bias=False)
            self.V = tf.keras.layers.Dense(1, use_bias=False)

        def call(self, inputs):
            score = self.V(tf.nn.tanh(self.W(inputs)))
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
            return context_vector

        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units})
            return config

    # ── Inline fallback: FocalLoss ──
    class FocalLoss(tf.keras.losses.Loss):
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
                    dtype=tf.float32
                )
            elif isinstance(alpha, (list, tuple)):
                self._alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            else:
                self._alpha_tensor = float(alpha)

        def call(self, y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            ce = -y_true * tf.math.log(y_pred)
            pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
            focal = tf.math.pow(1.0 - pt, self.gamma) * ce
            if isinstance(self._alpha_tensor, float):
                weighted = self._alpha_tensor * focal
            else:
                alpha_t = tf.reshape(self._alpha_tensor, (1, -1))
                weighted = alpha_t * y_true * focal
            return tf.reduce_mean(tf.reduce_sum(weighted, axis=-1))

        def get_config(self):
            config = super().get_config()
            config.update({"gamma": self.gamma, "alpha": self.alpha,
                           "class_weights": self.class_weights})
            return config

IMG_SIZE = CLASSIFIER_IMG_SIZE if 'CLASSIFIER_IMG_SIZE' in dir() else 224
OCR_CONFIDENCE_THRESHOLD = 0.3

# Number of spending categories the forecaster was trained on (NB04 v2).
# NB04 v2 includes Transport: N_FEATURES = 9, scaler.n_features_in_ = 9, Dense(9).
_FORECASTER_N_FEATURES = len(LABELS)  # 9


def assess_image_quality(image):
    """Assess image quality using Laplacian variance for blur detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    threshold = BLUR_THRESHOLD if 'BLUR_THRESHOLD' in dir() else 100
    is_blurry = blur_score < threshold
    return {"blur_score": float(blur_score), "is_blurry": bool(is_blurry)}


def clean_ocr_text(text):
    """Aggressive text cleaning: remove OCR noise, barcodes, normalize."""
    if not text or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'\b\d{8,}\b', '', text)
    text = re.sub(r'[^\w\s.,;:!?/\-()%Rr]', ' ', text)
    text = re.sub(r'\b[^0-9a-zA-Z]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class SpendlyOCR:
    """CRNN OCR model for receipt text extraction.
    Architecture synced with NB05 v6: Conv(32→64→128→128) + BiLSTM(256→128).
    Input: 50×400 grayscale.
    """
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.join(PROJECT_ROOT, 'models', 'ocr', 'ocr_weights_fixed.h5')
        # Character set MUST match NB05 training
        self.chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,:-/()%"
        self.idx_to_char = {i+1: c for i, c in enumerate(self.chars)}
        self.num_chars = len(self.chars) + 2  # +1 padding, +1 CTC blank
        self.model = self._build()
        self.model.load_weights(weights_path)

    def _build(self):
        """Build CRNN v6 — MUST match NB05 v6 architecture exactly."""
        from tensorflow.keras import layers, Model
        inputs = tf.keras.Input(shape=(50, 400, 1))

        # CNN Block 1: 50×400 → 25×200
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # CNN Block 2: 25×200 → 12×100
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # CNN Block 3: 12×100 → 6×100
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)

        # CNN Block 4 (v6 addition): 6×100 → 6×100
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout2D(0.15)(x)

        # Reshape for RNN: (batch, 6, 100, 128) → (batch, 100, 768)
        x = layers.Permute((2, 1, 3))(x)
        x = layers.Reshape((-1, x.shape[2] * x.shape[3]))(x)
        x = layers.Dropout(0.25)(x)

        # RNN — wider BiLSTM
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)

        # Dense head
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.25)(x)
        outputs = layers.Dense(self.num_chars, name='dense_out')(x)
        outputs = layers.Activation('linear', dtype='float32')(outputs)

        return Model(inputs, outputs)

    def predict(self, image_array):
        """Input: grayscale image (H, W) or (H, W, 1) or BGR (H, W, 3).
        Returns decoded text using unified preprocessing.
        """
        # Use unified preprocessing
        try:
            processed = preprocess_for_ocr(image_array, target_w=400, target_h=50)
        except:
            # Fallback if preprocess_for_ocr not available
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                img = image_array
            img = cv2.resize(img, (400, 50))
            if img.ndim == 2:
                img = np.expand_dims(img, -1)
            processed = img.astype(np.float32) / 255.0

        pred = self.model(np.expand_dims(processed, 0), training=False)[0]
        indices = tf.argmax(pred, axis=-1).numpy()
        text, prev = [], -1
        for idx in indices:
            if idx != 0 and idx != prev and idx in self.idx_to_char:
                text.append(self.idx_to_char[idx])
            prev = idx
        return ''.join(text)


class SpendlyTextClassifier:
    """Multimodal (TF-IDF + Deep CNN + SE) classifier.
    Architecture synced with NB03 v4.
    Loads the full .keras model (no separate .h5 weights file).
    """
    def __init__(self, keras_path=None, tfidf_path=None):
        if keras_path is None:
            keras_path = os.path.join(PROJECT_ROOT, 'models', 'classifier', 'classifier.keras')
        if tfidf_path is None:
            tfidf_path = os.path.join(PROJECT_ROOT, 'models', 'classifier', 'tfidf_vectorizer.joblib')
        self.tfidf = joblib.load(tfidf_path)
        self.model = tf.keras.models.load_model(
            keras_path,
            custom_objects={'SEBlock': SEBlock, 'FocalLoss': FocalLoss}
        )

    def predict(self, text, image_array=None):
        vec = self.tfidf.transform([text]).toarray().astype(np.float32)
        if image_array is not None:
            img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) if len(image_array.shape) == 3 else image_array
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
        else:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

        img_batch = np.expand_dims(img, 0)
        probs = self.model([vec, img_batch], training=False)[0].numpy()
        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx]), {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


class SpendlyForecaster:
    """LSTM + Attention spending forecaster.
    Architecture synced with NB04 v2: LSTM(64→32) + Attention.
    Trained on 9 categories (including Transport).
    """
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.join(PROJECT_ROOT, 'models', 'forecaster', 'forecaster_weights.h5')
        self.n_features = _FORECASTER_N_FEATURES  # 9 categories
        self.model = self._build()
        self.model.load_weights(weights_path)

    def _build(self):
        from tensorflow.keras import layers

        class AttentionLayerLocal(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            def build(self, input_shape):
                self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]),
                                         initializer='glorot_uniform', trainable=True)
                self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],),
                                         initializer='zeros', trainable=True)
                self.u = self.add_weight(name='att_context', shape=(input_shape[-1],),
                                         initializer='glorot_uniform', trainable=True)
            def call(self, x):
                score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
                attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
                return tf.reduce_sum(x * tf.expand_dims(attention_weights, -1), axis=1)
            def get_config(self):
                return super().get_config()

        inputs = tf.keras.Input(shape=(12, self.n_features))
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.LSTM(32, return_sequences=True)(x)
        x = AttentionLayerLocal(name='attention')(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.n_features)(x)
        return tf.keras.Model(inputs, outputs)

    def predict(self, recent_data):
        """Predict next-week spending for 9 categories.

        Args:
            recent_data: array of shape (12, 9) — last 12 weeks × 9 categories.
                         Column order must match LABELS.

        Returns:
            dict mapping all 9 LABELS to predicted values.
        """
        data = np.array(recent_data, dtype=np.float32)
        if data.ndim == 2:
            data = np.expand_dims(data, 0)
        pred = self.model(data, training=False)[0].numpy()
        result = {LABELS[i]: float(pred[i]) for i in range(self.n_features)}
        return result


class SpendlyReceiptDetector:
    """YOLOv8-style receipt/object detector.
    Loads the full .keras model from models/receipt_detector_best.keras.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'models', 'receipt_detector_best.keras')
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image_array, confidence_threshold=0.5):
        """Run detection on an input image.

        Args:
            image_array: BGR image as numpy array (H, W, 3).
            confidence_threshold: minimum confidence to keep a detection.

        Returns:
            list of dicts with keys: 'bbox' (x1,y1,x2,y2), 'confidence', 'class'.
        """
        original_h, original_w = image_array.shape[:2]

        # Preprocess: resize to model's expected input
        input_shape = self.model.input_shape  # e.g. (None, H, W, 3)
        target_h = input_shape[1] if input_shape[1] is not None else 640
        target_w = input_shape[2] if input_shape[2] is not None else 640

        img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h))
        img = img.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img, 0)

        raw_output = self.model(img_batch, training=False)
        if isinstance(raw_output, (list, tuple)):
            raw_output = raw_output[0]
        preds = raw_output.numpy()

        detections = []
        if preds.ndim == 3:
            # Shape (1, N, 5+) — each row: [x1, y1, x2, y2, conf, ...]
            for det in preds[0]:
                conf = float(det[4])
                if conf >= confidence_threshold:
                    x1 = float(det[0]) * original_w / target_w
                    y1 = float(det[1]) * original_h / target_h
                    x2 = float(det[2]) * original_w / target_w
                    y2 = float(det[3]) * original_h / target_h
                    cls_id = int(np.argmax(det[5:])) if len(det) > 5 else 0
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': cls_id
                    })
        elif preds.ndim == 2:
            # Shape (1, N) — binary classification output
            conf = float(preds[0][0]) if preds.shape[-1] == 1 else float(np.max(preds[0]))
            if conf >= confidence_threshold:
                detections.append({
                    'bbox': [0, 0, original_w, original_h],
                    'confidence': conf,
                    'class': 0
                })

        return detections
