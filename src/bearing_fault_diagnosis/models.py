"""Model definitions for bearing fault diagnosis experiments."""

from tensorflow import keras
from tensorflow.keras import layers


class CNNResNetModel:
    """Hybrid 1D CNN + residual network classifier."""

    def __init__(self, label_count, num_blocks, data_shape=(1000, 2)):
        self.num_blocks = num_blocks
        self.filters = 64
        self.kernel_size = 3
        self.label_count = label_count
        self.data_shape = data_shape
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = ["acc"]

    def _residual_block(self, inputs):
        x = layers.Conv1D(
            self.filters,
            self.kernel_size,
            activation="relu",
            padding="same",
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(
            self.filters,
            self.kernel_size,
            activation=None,
            padding="same",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, inputs])
        return layers.Activation("relu")(x)

    def build(self, learning_rate):
        inputs = keras.Input(shape=self.data_shape)
        x = layers.Conv1D(32, 3, activation="relu")(inputs)
        x = layers.Conv1D(64, 3, activation="relu")(x)
        x = layers.MaxPooling1D(16)(x)

        for _ in range(self.num_blocks):
            x = self._residual_block(x)

        x = layers.Conv1D(64, 3, activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.label_count, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=self.loss,
            metrics=self.metrics,
        )
        return model
