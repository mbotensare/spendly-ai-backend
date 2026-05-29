"""
Spendly AI — Custom Components
================================
Custom TensorFlow/Keras components untuk Spendly AI project.
- CTCLayer         : CTC loss layer untuk OCR model
- AttentionLayer   : Self-attention layer untuk sequence modeling
- SEBlock          : Squeeze-and-Excitation block (custom layer) untuk CNN branch
- FocalLoss        : Focal loss dengan per-class alpha untuk class imbalance
- SpendlyCallback  : Smart early stopping + model saving (support manual GradientTape loop)
- NLPPreprocessor  : NLP pipeline Bahasa Indonesia

Coding Camp 2026 powered by DBS Foundation (CC26-PSU276)
"""

import os
import re
import tensorflow as tf

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CTCLayer
# ─────────────────────────────────────────────────────────────────────────────
class CTCLayer(tf.keras.layers.Layer):
    """Custom CTC loss layer untuk OCR model.

    Menghitung CTC (Connectionist Temporal Classification) loss
    yang digunakan untuk sequence-to-sequence learning tanpa
    alignment eksplisit antara input dan output.
    """
    def __init__(self, name="ctc_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len    = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────────────────────────────────────
# AttentionLayer
# ─────────────────────────────────────────────────────────────────────────────
class AttentionLayer(tf.keras.layers.Layer):
    """Custom Attention Layer untuk sequence modeling.

    Menghitung self-attention weights terhadap sequence output,
    lalu menghasilkan context vector sebagai weighted sum.
    Cocok dipakai setelah LSTM/GRU layer.

    Args:
        units: Dimensi hidden state attention.
    Input shape : (batch, timesteps, features)
    Output shape: (batch, features)
    """
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1,     use_bias=False)

    def call(self, inputs):
        score            = self.V(tf.nn.tanh(self.W(inputs)))        # (batch, T, 1)
        attention_weights = tf.nn.softmax(score, axis=1)             # (batch, T, 1)
        context_vector   = tf.reduce_sum(attention_weights * inputs, axis=1)  # (batch, F)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ─────────────────────────────────────────────────────────────────────────────
# SEBlock  ← NEW custom layer
# ─────────────────────────────────────────────────────────────────────────────
class SEBlock(tf.keras.layers.Layer):
    """Squeeze-and-Excitation Block (custom layer).

    Secara adaptif merekalibrate channel-wise feature responses dengan
    memodelkan interdependensi antar channel. Digunakan di dalam CNN branch
    classifier untuk memperkuat channel yang relevan tanpa pretrained weights.

    Args:
        ratio: Reduction ratio untuk bottleneck. Default 8.

    Input shape : (batch, H, W, C)
    Output shape: (batch, H, W, C)  — sama dengan input, tapi channel di-rescale
    """
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels     = input_shape[-1]
        self.gap     = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1  = tf.keras.layers.Dense(
            max(1, channels // self.ratio), activation='relu', use_bias=False
        )
        self.dense2  = tf.keras.layers.Dense(channels, activation='sigmoid', use_bias=False)
        self.reshape = tf.keras.layers.Reshape((1, 1, channels))
        super().build(input_shape)

    def call(self, inputs):
        # Squeeze
        x = self.gap(inputs)            # (batch, C)
        # Excitation
        x = self.dense1(x)              # (batch, C//ratio)
        x = self.dense2(x)              # (batch, C)
        x = self.reshape(x)             # (batch, 1, 1, C)
        # Scale
        return inputs * x               # (batch, H, W, C)

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


# ─────────────────────────────────────────────────────────────────────────────
# FocalLoss  — updated: support per-class alpha
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss dengan dukungan per-class alpha untuk class imbalance.

    Args:
        gamma       : Focusing parameter. Semakin tinggi → fokus ke hard examples.
        alpha       : Weighting factor. Bisa scalar (float) atau list/array
                      per-class dengan panjang = num_classes.
        class_weights: Optional dict {class_idx: weight} dari sklearn
                       compute_class_weight. Jika diberikan, override alpha.
    """
    def __init__(self, gamma=2.0, alpha=0.25, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma        = gamma
        self.alpha        = alpha
        self.class_weights = class_weights

        # Resolusi alpha: class_weights > alpha list > alpha scalar
        if class_weights is not None:
            # Normalisasi ke [0,1] agar tidak meledakkan loss
            vals  = list(class_weights.values())
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
        ce     = -y_true * tf.math.log(y_pred)                          # (batch, C)
        pt     = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True) # (batch, 1)
        focal  = tf.math.pow(1.0 - pt, self.gamma) * ce                 # (batch, C)

        if isinstance(self._alpha_tensor, float):
            weighted = self._alpha_tensor * focal
        else:
            # broadcast per-class alpha: (C,) → (1, C)
            alpha_t  = tf.reshape(self._alpha_tensor, (1, -1))
            weighted = alpha_t * y_true * focal

        return tf.reduce_mean(tf.reduce_sum(weighted, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha,
                        "class_weights": self.class_weights})
        return config


# ─────────────────────────────────────────────────────────────────────────────
# SpendlyCallback — updated: tambah on_epoch_end_manual() untuk GradientTape
# ─────────────────────────────────────────────────────────────────────────────
class SpendlyCallback(tf.keras.callbacks.Callback):
    """Custom callback: log metrik, early stop cerdas, simpan best model.

    Support dua mode:
    - mode='fit'    : pakai on_epoch_end() standar Keras (model.fit)
    - mode='manual' : panggil on_epoch_end_manual(epoch, logs) dari
                      tf.GradientTape loop secara eksplisit

    Args:
        model_name : nama model, dipakai sebagai subfolder save path
        patience   : jumlah epoch tanpa improvement sebelum early stop
        min_delta  : minimum improvement yang dianggap signifikan
        save_dir   : direktori root untuk menyimpan model
        mode       : 'fit' atau 'manual'
    """
    def __init__(self, model_name, patience=10, min_delta=0.001,
                 save_dir="models", mode='manual'):
        super().__init__()
        self.model_name  = model_name
        self.patience    = patience
        self.min_delta   = min_delta
        self.save_dir    = save_dir
        self.mode        = mode
        self.best_val_acc = 0.0
        self.wait        = 0
        self.best_epoch  = 0
        self.stop_training = False   # flag untuk manual loop

    def _process(self, epoch, logs):
        """Logic inti: print, save, early-stop check."""
        logs    = logs or {}
        val_acc  = logs.get("val_accuracy", logs.get("val_acc", 0.0))
        val_loss = logs.get("val_loss", 0.0)
        trn_acc  = logs.get("accuracy",   logs.get("train_acc", 0.0))
        trn_loss = logs.get("loss",       logs.get("train_loss", 0.0))

        marker = ""
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.best_epoch   = epoch + 1
            self.wait         = 0
            save_path = os.path.join(self.save_dir, self.model_name,
                                     f"{self.model_name}.keras")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            marker = f"  ← BEST saved ({save_path})"
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"[SpendlyCallback] Early stop di epoch {epoch+1}. "
                      f"Best epoch: {self.best_epoch} "
                      f"(val_acc: {self.best_val_acc:.4f})")
                self.stop_training = True
                # juga set flag Keras agar model.fit aware
                if self.model is not None:
                    self.model.stop_training = True

        print(f"[SpendlyCallback] Epoch {epoch+1:3d} | "
              f"loss={trn_loss:.4f} acc={trn_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"best={self.best_val_acc:.4f} patience={self.wait}/{self.patience}"
              f"{marker}")

    # ── mode='fit' ──────────────────────────────────────────────────────────
    def on_epoch_end(self, epoch, logs=None):
        if self.mode == 'fit':
            self._process(epoch, logs)

    # ── mode='manual' ───────────────────────────────────────────────────────
    def on_epoch_end_manual(self, epoch, logs=None):
        """Panggil ini secara eksplisit dari tf.GradientTape loop.

        Returns:
            bool: True jika training harus dihentikan (early stop).
        """
        self._process(epoch, logs)
        return self.stop_training


# ─────────────────────────────────────────────────────────────────────────────
# NLPPreprocessor
# ─────────────────────────────────────────────────────────────────────────────
class NLPPreprocessor:
    """Pipeline NLP untuk teks transaksi Bahasa Indonesia."""
    def __init__(self):
        if SASTRAWI_AVAILABLE:
            self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
            self.stemmer          = StemmerFactory().create_stemmer()
        else:
            self.stopword_remover = None
            self.stemmer          = None

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ''
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\d+', 'NUM', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if self.stopword_remover:
            text = self.stopword_remover.remove(text)
        if self.stemmer:
            text = self.stemmer.stem(text)
        return text

    def preprocess_batch(self, texts):
        return [self.preprocess(t) for t in texts]
