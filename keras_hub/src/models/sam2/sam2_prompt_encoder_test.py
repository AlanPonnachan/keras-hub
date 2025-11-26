import numpy as np
from keras import ops

from keras_hub.src.models.sam2.sam2_prompt_encoder import SAMPromptEncoder
from keras_hub.src.tests.test_case import TestCase


class SAMPromptEncoderTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = 32
        self.hidden_size = 16
        self.image_embedding_size = (4, 4)
        self.mask_in_channels = 4

        self.init_kwargs = {
            "hidden_size": self.hidden_size,
            "image_embedding_size": self.image_embedding_size,
            "input_image_size": (self.image_size, self.image_size),
            "mask_in_channels": self.mask_in_channels,
        }
        self.prompt_encoder = SAMPromptEncoder(**self.init_kwargs)

    def test_layer_basics(self):
        self.run_layer_test(
            cls=SAMPromptEncoder,
            init_kwargs=self.init_kwargs,
            input_data={
                "points": np.ones((2, 5, 2), dtype="float32"),
                "labels": np.ones((2, 5), dtype="int32"),
                "boxes": np.ones((2, 2, 2, 2), dtype="float32"),
                "masks": np.ones((2, 16, 16, 1), dtype="float32"),
            },
            expected_output_shape={
                "prompt_sparse_embeddings": (2, 9, 16),
                "prompt_dense_embeddings": (2, 4, 4, 16),
                "prompt_dense_positional_embeddings": (2, 4, 4, 16),
            },
            expected_num_trainable_weights=16,
            expected_num_non_trainable_weights=1,
            expected_num_non_trainable_variables=1,
        )

    def test_forward_padding_logic(self):
        points = np.zeros((1, 5, 2), dtype="float32")
        labels = np.zeros((1, 5), dtype="int32")
        outputs = self.prompt_encoder(points=points, labels=labels)
        sparse = outputs["prompt_sparse_embeddings"]
        self.assertEqual(sparse.shape, (1, 6, 16))

        boxes = np.zeros((1, 2, 2, 2), dtype="float32")
        outputs = self.prompt_encoder(points=points, labels=labels, boxes=boxes)
        sparse = outputs["prompt_sparse_embeddings"]

        self.assertEqual(sparse.shape, (1, 9, 16))

        outputs = self.prompt_encoder(boxes=boxes)
        sparse = outputs["prompt_sparse_embeddings"]

        self.assertEqual(sparse.shape, (1, 4, 16))

    def test_dense_embeddings_logic(self):
        boxes = np.zeros((1, 1, 2, 2), dtype="float32")
        outputs = self.prompt_encoder(boxes=boxes)
        dense = outputs["prompt_dense_embeddings"]
        self.assertEqual(dense.shape, (1, 4, 4, 16))

        dense_np = ops.convert_to_numpy(dense)
        self.assertTrue(np.all(dense_np[0, 0, 0, :] == dense_np[0, 1, 1, :]))

        masks = np.random.rand(1, 16, 16, 1).astype("float32")
        outputs = self.prompt_encoder(masks=masks)
        dense = outputs["prompt_dense_embeddings"]
        self.assertEqual(dense.shape, (1, 4, 4, 16))
