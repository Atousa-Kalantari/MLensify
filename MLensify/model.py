import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import ops
import joblib
import os
import random as rn
from keras.saving import register_keras_serializable




# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
rn.seed(SEED)

# Define constants
ORIGINAL_DIM = 940
PADDED_DIM = 960
LATENT_DIM = 20

# --- Redefine Custom Layers ---
@register_keras_serializable()
class Sampling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(SEED)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class GlobalMinPooling1D(Layer):
    def call(self, inputs):
        return tf.reduce_min(inputs, axis=1)

class GlobalStdPooling1D(Layer):
    def call(self, inputs):
        return tf.math.reduce_std(inputs, axis=1)

# --- Redefine CVAE Class ---

class CVAE(keras.Model):
    def __init__(self, encoder, decoder, alpha=None, beta=None, rec_weight=1.0, kl_weight=1.0, beta_loss_weight=0.0, fwhm_loss_weight=1.0, classification_weight=20.0, normalize_rec=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        self.beta = beta
        self.rec_weight = rec_weight
        self.kl_weight = kl_weight
        self.beta_loss_weight = beta_loss_weight
        self.fwhm_loss_weight = fwhm_loss_weight
        self.classification_weight = classification_weight
        self.normalize_rec = normalize_rec
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.fwhm_loss_tracker = keras.metrics.Mean(name="fwhm_loss")
        self.classification_loss_tracker = keras.metrics.Mean(name="classification_loss")
        self.accuracy_tracker = keras.metrics.BinaryAccuracy(name="accuracy")
        self.bce_loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError(reduction="none")

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
            'rec_weight': self.rec_weight,
            'kl_weight': self.kl_weight,
            'beta_loss_weight': self.beta_loss_weight,
            'fwhm_loss_weight': self.fwhm_loss_weight,
            'classification_weight': self.classification_weight,
            'normalize_rec': self.normalize_rec,
            'encoder_config': self.encoder.get_config(),
            'decoder_config': self.decoder.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        encoder_config = config.pop('encoder_config')
        decoder_config = config.pop('decoder_config')
        encoder = tf.keras.models.Model.from_config(encoder_config, custom_objects=custom_objects)
        decoder = tf.keras.models.Model.from_config(decoder_config, custom_objects=custom_objects)
        return cls(encoder=encoder, decoder=decoder, **config)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.fwhm_loss_tracker,
            self.classification_loss_tracker,
            self.accuracy_tracker,
        ]

    def _calculate_loss(self, data_all):
        data = data_all[:, :PADDED_DIM, :]
        scalar_features = data_all[:, PADDED_DIM:PADDED_DIM+10, 0]
        fwhm = data_all[:, PADDED_DIM+10, 0]
        y_true = data_all[:, PADDED_DIM+11, 0]

        z_mean, z_log_var, z, class_pred = self.encoder([data, scalar_features])
        reconstruction = self.decoder([z, scalar_features])
        original_data = data[:, :ORIGINAL_DIM, :]

        bce_loss_per_sample = ops.sum(self.bce_loss_fn(original_data, reconstruction), axis=1)
        mse_loss_per_sample = ops.sum(self.mse_loss_fn(original_data, reconstruction), axis=1)
        combined_loss_per_sample = self.alpha * bce_loss_per_sample + self.beta * mse_loss_per_sample
        reconstruction_loss = ops.mean(combined_loss_per_sample)
        if self.normalize_rec:
            reconstruction_loss /= ORIGINAL_DIM

        kl_loss_per_sample = -0.5 * ops.sum(1 + z_log_var[:, :-2] - ops.square(z_mean[:, :-2]) - ops.exp(z_log_var[:, :-2]), axis=1)
        kl_loss = ops.mean(kl_loss_per_sample)

        fwhm_loss = 0.0
        classification_loss = 0.0

        if self.classification_weight > 0:
            classification_loss_per_sample = self.bce_loss_fn(y_true, class_pred)
            classification_loss = ops.mean(classification_loss_per_sample)

        if self.fwhm_loss_weight > 0:
            std = ops.exp(0.5 * z_log_var[:, -2])
            var = ops.exp(z_log_var[:, -2])
            tE_std = 0.05
            tE_var = tE_std ** 2
            kl_tE = (
                ops.log(tE_std / std)
                + (var + ops.square(z_mean[:, -2] - fwhm)) / (2 * tE_var)
                - 0.5
            )
            fwhm_loss = ops.mean(kl_tE)

        total_loss = (self.rec_weight * reconstruction_loss) + \
                     (self.kl_weight * kl_loss) + \
                     (self.fwhm_loss_weight * fwhm_loss) + \
                     (self.classification_weight * classification_loss)

        return total_loss, reconstruction_loss, kl_loss, fwhm_loss, classification_loss, class_pred, y_true

    def train_step(self, data_all):
        with tf.GradientTape() as tape:
            total_loss, rec_loss, kl_loss, fwhm_loss, class_loss, class_pred, y_true = self._calculate_loss(data_all)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.fwhm_loss_tracker.update_state(fwhm_loss)
        self.classification_loss_tracker.update_state(class_loss)
        self.accuracy_tracker.update_state(y_true, class_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data_all):
        total_loss, rec_loss, kl_loss, fwhm_loss, class_loss, class_pred, y_true = self._calculate_loss(data_all)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.fwhm_loss_tracker.update_state(fwhm_loss)
        self.classification_loss_tracker.update_state(class_loss)
        self.accuracy_tracker.update_state(y_true, class_pred)
        return {m.name: m.result() for m in self.metrics}

