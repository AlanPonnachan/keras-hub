from keras import random

from keras_hub.src.models.sam2.sam2_memory_attention import SAM2MemoryAttention
from keras_hub.src.tests.test_case import TestCase


class SAM2MemoryAttentionTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.d_model = 16
        self.num_heads = 2
        self.num_layers = 1
        self.feat_size = 4

        self.init_kwargs = {
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dim_feedforward": 32,
            "rope_feat_sizes": (self.feat_size, self.feat_size),
        }

        self.input_data = {
            "curr": random.uniform(shape=(self.batch_size, 16, self.d_model)),
            "memory": random.uniform(shape=(self.batch_size, 32, self.d_model)),
            "curr_pos": random.uniform(
                shape=(self.batch_size, 16, self.d_model)
            ),
            "memory_pos": random.uniform(
                shape=(self.batch_size, 32, self.d_model)
            ),
        }

    def test_layer_basics(self):
        layer = SAM2MemoryAttention(**self.init_kwargs)

        outputs = layer(**self.input_data)
        expected_shape = (self.batch_size, 16, self.d_model)
        self.assertEqual(outputs.shape, expected_shape)

        self.assertEqual(len(layer.trainable_weights), 28)

    def test_serialization(self):
        layer = SAM2MemoryAttention(**self.init_kwargs)
        config = layer.get_config()
        for key in self.init_kwargs:
            self.assertEqual(config[key], self.init_kwargs[key])
