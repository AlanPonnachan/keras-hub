from keras import random

from keras_hub.src.models.sam2.sam2_memory_encoder import SAM2MemoryEncoder
from keras_hub.src.tests.test_case import TestCase


class SAM2MemoryEncoderTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_dim = 16
        self.out_dim = 8
        self.image_size = 32

        self.init_kwargs = {
            "out_dim": self.out_dim,
            "in_dim": self.in_dim,
            "mask_downsampler_kernel_size": 3,
            "mask_downsampler_stride": 2,
            "mask_downsampler_total_stride": 4,
            "fuser_num_layers": 1,
            "fuser_dim": self.in_dim,
            "activation": "relu",
        }

        self.input_data = {
            "pix_feat": random.uniform(
                shape=(
                    self.batch_size,
                    self.image_size,
                    self.image_size,
                    self.in_dim,
                )
            ),
            "masks": random.uniform(
                shape=(
                    self.batch_size,
                    self.image_size * 4,
                    self.image_size * 4,
                    1,
                )
            ),
        }

    def test_layer_basics(self):
        self.run_layer_test(
            cls=SAM2MemoryEncoder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "vision_features": (
                    self.batch_size,
                    self.image_size,
                    self.image_size,
                    self.out_dim,
                ),
                "vision_pos_enc": (
                    self.batch_size,
                    self.image_size,
                    self.image_size,
                    self.out_dim,
                ),
            },
            expected_num_trainable_weights=23,
        )
