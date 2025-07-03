# custom_layers.py
import tensorflow as tf

# ----- Squeeze-and-Excitation (SE) Block -----
def se_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]
    x = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    x = tf.keras.layers.Dense(channels // reduction, activation='relu')(x)
    x = tf.keras.layers.Dense(channels, activation='sigmoid')(x)
    x = tf.keras.layers.Reshape((1, 1, channels))(x)
    return tf.keras.layers.Multiply()([input_tensor, x])

# ----- CBAM Block -----
class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        self.channels = input_shape[-1]

        # Shared MLP for Channel Attention
        self.shared_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.channels // self.reduction, activation='relu'),
            tf.keras.layers.Dense(self.channels)
        ])

        # Spatial Attention convolution
        self.spatial_conv = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        # ----- Channel Attention -----
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)

        avg_out = self.shared_dense(avg_pool)
        max_out = self.shared_dense(max_pool)

        channel = tf.keras.layers.Add()([avg_out, max_out])
        channel = tf.keras.layers.Activation('sigmoid')(channel)
        channel = tf.keras.layers.Reshape((1, 1, self.channels))(channel)

        x = tf.keras.layers.Multiply()([inputs, channel])

        # ----- Spatial Attention -----
        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial = tf.keras.layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        spatial = self.spatial_conv(spatial)

        return tf.keras.layers.Multiply()([x, spatial])
