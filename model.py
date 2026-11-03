import tensorflow as tf
from tensorflow.keras import layers, models

## Model definition
def create_cdae_model(T, R):
    input_shape = (T, R, 2)  # Input is (T, R, 2), where 2 represents real + imaginary components

    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=((0, T % 8), (0, R % 8)))(inputs)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(16, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(4, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    z = layers.Dense(128, name="feature_vector")(x)

    # Decoder
    x = layers.Dense(4 * (T // 8) * (R // 8))(z)
    x = layers.Reshape((T // 8, R // 8, 4))(x)

    x = layers.Conv2DTranspose(16, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D(size=(2, 2))(x)

    x = layers.Conv2DTranspose(32, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D(size=(2, 2))(x)

    x = layers.Conv2DTranspose(32, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D(size=(2, 2))(x)

    outputs = layers.Conv2DTranspose(2, kernel_size=(3, 3), padding="same")(x)

    model = models.Model(inputs, outputs)
    return model

# PSNR metric for evaluation
def psnr_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    max_val = tf.reduce_max(y_true)
    psnr = 10.0 * tf.math.log((max_val**2) / (mse + 1e-8)) / tf.math.log(10.0)
    return psnr

# # SSIM metric for evaluation
# def ssim_metric(y_true, y_pred):
#     y_true = tf.cast(y_true, tf.float32)  # Ensure both are float32
#     y_pred = tf.cast(y_pred, tf.float32)
#     return tf.image.ssim(y_true, y_pred, max_val=1.0)