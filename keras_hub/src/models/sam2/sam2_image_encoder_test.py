from keras import random

from keras_hub.src.models.sam2.sam2_image_encoder import SAM2ImageEncoder
from keras_hub.src.tests.test_case import TestCase


class SAM2ImageEncoderTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = 64

        self.init_kwargs = {
            "embed_dim": 16,
            "num_heads": 1,
            "stages": (1, 1, 1, 1),
            "global_att_blocks": (2,),
            "window_pos_embed_bkg_spatial_size": (4, 4),
            "window_spec": (8, 4, 4, 2),
            "d_model": 16,
            "backbone_channel_list": [16, 32, 64, 128],
            "scalp": 1,
        }

        self.input_data = random.uniform(
            minval=0,
            maxval=1,
            shape=(self.batch_size, self.image_size, self.image_size, 3),
        )

    def test_layer_basics(self):
        self.run_layer_test(
            cls=SAM2ImageEncoder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "vision_features": (self.batch_size, 4, 4, 16),
                "vision_pos_enc": (self.batch_size, 4, 4, 16),
                # List of features for [stride 4, stride 8, stride 16]
                "backbone_fpn": [
                    (self.batch_size, 16, 16, 16),
                    (self.batch_size, 8, 8, 16),
                    (self.batch_size, 4, 4, 16),
                ],
                "backbone_pos_enc": [
                    (self.batch_size, 16, 16, 16),
                    (self.batch_size, 8, 8, 16),
                    (self.batch_size, 4, 4, 16),
                ],
            },
            expected_num_trainable_weights=66,
            run_training_check=False,
        )

    def test_forward_pass(self):
        layer = SAM2ImageEncoder(**self.init_kwargs)
        outputs = layer(self.input_data)

        self.assertEqual(len(outputs["backbone_fpn"]), 3)
        self.assertEqual(len(outputs["backbone_pos_enc"]), 3)

        self.assertEqual(
            outputs["backbone_fpn"][0].shape, (self.batch_size, 16, 16, 16)
        )
        self.assertEqual(
            outputs["backbone_fpn"][1].shape, (self.batch_size, 8, 8, 16)
        )
        self.assertEqual(
            outputs["backbone_fpn"][2].shape, (self.batch_size, 4, 4, 16)
        )

        self.assertEqual(
            outputs["vision_features"].shape, (self.batch_size, 4, 4, 16)
        )
