from keras import random

from keras_hub.src.models.sam2.sam2_mask_decoder import SAM2MaskDecoder
from keras_hub.src.tests.test_case import TestCase


class SAM2MaskDecoderTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.embedding_dim = 16
        self.hidden_size = 16
        self.image_size = 8

        self.init_kwargs = {
            "num_layers": 2,
            "hidden_size": self.hidden_size,
            "intermediate_dim": 32,
            "num_heads": 2,
            "embedding_dim": self.embedding_dim,
            "num_multimask_outputs": 3,
            "iou_head_depth": 3,
            "iou_head_hidden_dim": 16,
            "activation": "relu",
        }

        self.input_data = {
            "image_embeddings": random.uniform(
                minval=0,
                maxval=1,
                shape=(
                    self.batch_size,
                    self.image_size,
                    self.image_size,
                    self.embedding_dim,
                ),
            ),
            "image_pe": random.uniform(
                minval=0,
                maxval=1,
                shape=(
                    self.batch_size,
                    self.image_size,
                    self.image_size,
                    self.embedding_dim,
                ),
            ),
            "sparse_prompt_embeddings": random.uniform(
                minval=0,
                maxval=1,
                shape=(self.batch_size, 5, self.embedding_dim),
            ),
            "dense_prompt_embeddings": random.uniform(
                minval=0,
                maxval=1,
                shape=(
                    self.batch_size,
                    self.image_size,
                    self.image_size,
                    self.embedding_dim,
                ),
            ),
        }

    def test_layer_basics(self):
        self.run_layer_test(
            cls=SAM2MaskDecoder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "masks": (
                    self.batch_size,
                    4,
                    self.image_size * 4,
                    self.image_size * 4,
                ),
                "iou_pred": (self.batch_size, 4),
                "sam_tokens": (self.batch_size, 4, self.embedding_dim),
                "object_score_logits": (self.batch_size, 1),
            },
            expected_num_trainable_weights=120,
        )

    def test_extended_config(self):
        kwargs = self.init_kwargs.copy()
        kwargs["pred_obj_scores"] = True
        kwargs["use_high_res_features"] = True

        layer = SAM2MaskDecoder(**kwargs)

        s1_shape = (
            self.batch_size,
            self.image_size * 2,
            self.image_size * 2,
            self.embedding_dim // 4,
        )
        s0_shape = (
            self.batch_size,
            self.image_size * 4,
            self.image_size * 4,
            self.embedding_dim // 8,
        )

        inputs = self.input_data.copy()
        inputs["high_res_features"] = [
            random.uniform(shape=s0_shape),
            random.uniform(shape=s1_shape),
        ]

        outputs = layer(**inputs)

        self.assertEqual(len(layer.trainable_weights), 127)

        self.assertEqual(
            outputs["masks"].shape,
            (self.batch_size, 4, self.image_size * 4, self.image_size * 4),
        )

        self.assertEqual(
            outputs["object_score_logits"].shape, (self.batch_size, 1)
        )
