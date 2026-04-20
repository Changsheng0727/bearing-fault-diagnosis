"""Model definitions for bearing fault diagnosis experiments."""

from tensorflow import keras
from tensorflow.keras import layers


class SimpleCNNModel:
    """Lightweight 1D CNN baseline that matches the original notebook design."""

    def __init__(
        self,
        label_count,
        data_shape=(1000, 2),
        conv_filters=64,
        dense_units=32,
        dropout_rate=0.5,
    ):
        self.label_count = label_count
        self.data_shape = data_shape
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = ["acc"]

    def build(self, learning_rate):
        inputs = keras.Input(shape=self.data_shape)
        x = layers.Conv1D(self.conv_filters, 3, activation="relu")(inputs)
        x = layers.MaxPooling1D(16)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(self.dense_units, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.label_count, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=self.loss,
            metrics=self.metrics,
        )
        return model


class CNNResNetModel:
    """Hybrid 1D CNN + residual network classifier."""

    def __init__(
        self,
        label_count,
        num_blocks,
        data_shape=(1000, 2),
        filters=64,
        kernel_size=3,
        stem_filters=(32, 64),
        dense_units=256,
        dropout_rate=0.5,
    ):
        self.num_blocks = num_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.stem_filters = stem_filters
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
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
        x = layers.Conv1D(
            self.stem_filters[0],
            self.kernel_size,
            activation="relu",
        )(inputs)
        x = layers.Conv1D(
            self.stem_filters[1],
            self.kernel_size,
            activation="relu",
        )(x)
        x = layers.MaxPooling1D(16)(x)

        for _ in range(self.num_blocks):
            x = self._residual_block(x)

        x = layers.Conv1D(
            self.filters,
            self.kernel_size,
            activation="relu",
        )(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(self.dense_units, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.label_count, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=self.loss,
            metrics=self.metrics,
        )
        return model
