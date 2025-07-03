# custom_layers.py
import tensorflow as tf

# ============================================================
# Squeeze-and-Excitation (SE) Block
# ------------------------------------------------------------
# Adds channel-wise attention by explicitly modeling interdependencies
# between channels through a squeeze (global average pooling) and excitation
# (fully connected layers) mechanism.
# ============================================================

def se_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]

    # Global context vector (squeeze)
    x = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)

    # Fully connected bottleneck (excitation)
    x = tf.keras.layers.Dense(channels // reduction, activation='relu')(x)
    x = tf.keras.layers.Dense(channels, activation='sigmoid')(x)

    # Reshape to match original dimensions and scale the input tensor
    x = tf.keras.layers.Reshape((1, 1, channels))(x)
    return tf.keras.layers.Multiply()([input_tensor, x])

# ============================================================
# Convolutional Block Attention Module (CBAM)
# ------------------------------------------------------------
# Combines both channel attention and spatial attention sequentially.
# CBAM improves representational power by focusing on 'what' and 'where'
# to attend in the feature maps.
# ============================================================

class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction = reduction  # Reduction ratio for channel attention MLP

    def build(self, input_shape):
        self.channels = input_shape[-1]

        # Shared MLP for both average and max pooled channel features
        self.shared_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.channels // self.reduction, activation='relu'),
            tf.keras.layers.Dense(self.channels)
        ])

        # Convolution layer to compute spatial attention map
        self.spatial_conv = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        # =====================
        # Channel Attention
        # =====================
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)  # Avg pooled features
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)      # Max pooled features

        # Pass through shared MLP
        avg_out = self.shared_dense(avg_pool)
        max_out = self.shared_dense(max_pool)

        # Combine and apply sigmoid to get channel attention map
        channel = tf.keras.layers.Add()([avg_out, max_out])
        channel = tf.keras.layers.Activation('sigmoid')(channel)
        channel = tf.keras.layers.Reshape((1, 1, self.channels))(channel)

        # Scale input with channel attention
        x = tf.keras.layers.Multiply()([inputs, channel])

        # =====================
        # Spatial Attention
        # =====================
        # Reduce along channel axis
        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)

        # Concatenate and pass through spatial conv
        spatial = tf.keras.layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        spatial = self.spatial_conv(spatial)

        # Final output: refined feature map
        return tf.keras.layers.Multiply()([x, spatial])
