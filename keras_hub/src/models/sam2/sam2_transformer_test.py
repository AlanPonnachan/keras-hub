from keras import random

from keras_hub.src.models.sam2.sam2_layers import (
    FixedFrequencyPositionalEmbeddings,
)
from keras_hub.src.models.sam2.sam2_layers import SAM2Attention
from keras_hub.src.models.sam2.sam2_transformer import TwoWayTransformer
from keras_hub.src.tests.test_case import TestCase


class FixedFrequencyPositionalEmbeddingsTest(TestCase):
    def test_layer_basics(self):
        self.run_layer_test(
            cls=FixedFrequencyPositionalEmbeddings,
            init_kwargs={
                "num_positional_features": 16,
                "temperature": 100,
                "normalize": True,
                "scale": 1.0,
            },
            input_data=random.uniform(shape=(2, 32, 32, 16)),
            expected_output_shape=(2, 32, 32, 16),
            expected_num_trainable_weights=0,
        )


class SAM2AttentionTest(TestCase):
    def test_layer_basics(self):
        self.run_layer_test(
            cls=SAM2Attention,
            init_kwargs={
                "num_heads": 2,
                "key_dim": 8,
                "embedding_dim": 16,
                "downsample_rate": 1,
            },
            input_data={
                "query": random.uniform(shape=(2, 10, 16)),
                "key": random.uniform(shape=(2, 20, 16)),
                "value": random.uniform(shape=(2, 20, 16)),
            },
            expected_output_shape=(2, 10, 16),
            expected_num_trainable_weights=8,
        )

    def test_downsampling(self):
        self.run_layer_test(
            cls=SAM2Attention,
            init_kwargs={
                "num_heads": 2,
                "key_dim": 4,
                "embedding_dim": 16,
                "downsample_rate": 2,
            },
            input_data={
                "query": random.uniform(shape=(2, 10, 16)),
                "key": random.uniform(shape=(2, 20, 16)),
                "value": random.uniform(shape=(2, 20, 16)),
            },
            expected_output_shape=(2, 10, 16),
            expected_num_trainable_weights=8,
        )


class TwoWayTransformerTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.embedding_dim = 16
        self.num_heads = 2
        self.mlp_dim = 32
        self.depth = 2
        self.image_size = 16
        self.num_points = 5

        self.init_kwargs = {
            "depth": self.depth,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "activation": "relu",
            "attention_downsample_rate": 2,
        }

        self.input_data = {
            "image_embedding": random.uniform(
                shape=(
                    self.batch_size,
                    self.image_size,
                    self.image_size,
                    self.embedding_dim,
                )
            ),
            "image_pe": random.uniform(
                shape=(
                    self.batch_size,
                    self.image_size,
                    self.image_size,
                    self.embedding_dim,
                )
            ),
            "point_embedding": random.uniform(
                shape=(self.batch_size, self.num_points, self.embedding_dim)
            ),
        }

    def test_forward_pass(self):
        layer = TwoWayTransformer(**self.init_kwargs)
        queries, keys = layer(**self.input_data)

        self.assertEqual(
            queries.shape,
            (self.batch_size, self.num_points, self.embedding_dim),
        )
        self.assertEqual(
            keys.shape,
            (
                self.batch_size,
                self.image_size * self.image_size,
                self.embedding_dim,
            ),
        )

        self.assertEqual(len(layer.trainable_weights), 82)

    def test_serialization(self):
        layer = TwoWayTransformer(**self.init_kwargs)
        config = layer.get_config()

        for key in self.init_kwargs:
            self.assertEqual(config[key], self.init_kwargs[key])
