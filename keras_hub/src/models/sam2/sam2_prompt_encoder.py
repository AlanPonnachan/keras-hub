import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.sam2.sam2_layers import (
    RandomFrequencyPositionalEmbeddings,
)


@keras_hub_export("keras_hub.models.SAMPromptEncoder")
class SAMPromptEncoder(keras.layers.Layer):
    """Prompt Encoder for the Segment Anything Model 2 (SAM2).

    The prompt encoder generates encodings for three types of prompts:
    - Point prompts: Points on the image along with a label indicating whether
        the point is in the foreground or background.
    - Box prompts: Bounding boxes.
    - Masks: Input masks.

    Args:
        hidden_size: int. The number of features in the output embeddings.
        image_embedding_size: tuple[int]. The spatial size of the image
            embeddings (H, W).
        input_image_size: tuple[int]. The spatial size of the input image
            (H, W).
        mask_in_channels: int. The number of channels of the mask prompt.
        activation: str. The activation to use in the mask downscaler.
    """

    def __init__(
        self,
        hidden_size,
        image_embedding_size,
        input_image_size,
        mask_in_channels,
        activation="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.mask_in_channels = mask_in_channels
        self.activation = activation

        self.pe_layer = RandomFrequencyPositionalEmbeddings(
            num_positional_features=self.hidden_size // 2,
            scale=1.0,
            dtype=self.dtype_policy,
        )

        self.point_embeddings = [
            keras.layers.Embedding(1, hidden_size, dtype=self.dtype_policy)
            for _ in range(4)
        ]
        self.not_a_point_embed = keras.layers.Embedding(
            1, hidden_size, dtype=self.dtype_policy
        )

        self.mask_downscaler = keras.models.Sequential(
            [
                keras.layers.Conv2D(
                    mask_in_channels // 4,
                    kernel_size=2,
                    strides=2,
                    dtype=self.dtype_policy,
                ),
                keras.layers.LayerNormalization(
                    epsilon=1e-6, dtype=self.dtype_policy
                ),
                keras.layers.Activation(activation, dtype=self.dtype_policy),
                keras.layers.Conv2D(
                    mask_in_channels,
                    kernel_size=2,
                    strides=2,
                    dtype=self.dtype_policy,
                ),
                keras.layers.LayerNormalization(
                    epsilon=1e-6, dtype=self.dtype_policy
                ),
                keras.layers.Activation(activation, dtype=self.dtype_policy),
                keras.layers.Conv2D(
                    hidden_size, kernel_size=1, dtype=self.dtype_policy
                ),
            ],
            name="mask_downscaler",
        )
        self.mask_downscaler.dtype_policy = self.dtype_policy

        self.no_mask_embed = keras.layers.Embedding(
            1, hidden_size, dtype=self.dtype_policy
        )

    def build(
        self,
        points_shape=None,
        labels_shape=None,
        boxes_shape=None,
        masks_shape=None,
    ):
        self.pe_layer.build(None)
        for layer in self.point_embeddings:
            layer.build(None)
        self.not_a_point_embed.build(None)
        self.no_mask_embed.build(None)

        downscaler_input_shape = (
            None,
            4 * self.image_embedding_size[0],
            4 * self.image_embedding_size[1],
            1,
        )
        self.mask_downscaler.build(downscaler_input_shape)
        self.built = True

    def _embed_points(self, points, labels, pad):
        """Embeds point prompts."""
        points = points + 0.5

        if pad:
            batch_size = ops.shape(points)[0]
            padding_point = ops.zeros((batch_size, 1, 2), dtype=points.dtype)
            padding_label = -1 * ops.ones((batch_size, 1), dtype=labels.dtype)

            points = ops.concatenate([points, padding_point], axis=1)
            labels = ops.concatenate([labels, padding_label], axis=1)

        point_embedding = self.pe_layer.encode_coordinates(
            points, self.input_image_size
        )

        indices = ops.zeros_like(labels)

        not_a_point_vec = self.not_a_point_embed(indices)
        point_vec_0 = self.point_embeddings[0](indices)
        point_vec_1 = self.point_embeddings[1](indices)
        point_vec_2 = self.point_embeddings[2](indices)
        point_vec_3 = self.point_embeddings[3](indices)

        labels_expanded = labels[..., None]

        point_embedding = ops.where(
            labels_expanded == -1,
            ops.zeros_like(point_embedding) + not_a_point_vec,
            point_embedding,
        )

        point_embedding = ops.where(
            labels_expanded == 0, point_embedding + point_vec_0, point_embedding
        )
        point_embedding = ops.where(
            labels_expanded == 1, point_embedding + point_vec_1, point_embedding
        )
        point_embedding = ops.where(
            labels_expanded == 2, point_embedding + point_vec_2, point_embedding
        )
        point_embedding = ops.where(
            labels_expanded == 3, point_embedding + point_vec_3, point_embedding
        )

        return point_embedding

    def _embed_boxes(self, boxes):
        """Embeds box prompts."""
        boxes = boxes + 0.5

        shape = ops.shape(boxes)
        batch_size, num_boxes = shape[0], shape[1]

        coords = ops.reshape(boxes, (batch_size, num_boxes * 2, 2))

        corner_embedding = self.pe_layer.encode_coordinates(
            coords, self.input_image_size
        )

        corner_embedding = ops.reshape(
            corner_embedding, (batch_size, num_boxes, 2, self.hidden_size)
        )

        indices = ops.zeros((batch_size, num_boxes), dtype="int32")
        top_left_vec = self.point_embeddings[2](indices)[:, :, None, :]
        bottom_right_vec = self.point_embeddings[3](indices)[:, :, None, :]

        adder = ops.concatenate([top_left_vec, bottom_right_vec], axis=2)
        corner_embedding = corner_embedding + adder

        return ops.reshape(
            corner_embedding, (batch_size, num_boxes * 2, self.hidden_size)
        )

    def _embed_masks(self, masks):
        """Embeds mask inputs."""
        return self.mask_downscaler(masks)

    def call(self, points=None, labels=None, boxes=None, masks=None):
        if points is not None:
            batch_size = ops.shape(points)[0]
        elif boxes is not None:
            batch_size = ops.shape(boxes)[0]
        elif masks is not None:
            batch_size = ops.shape(masks)[0]
        else:
            batch_size = 1

        sparse_embeddings = ops.zeros(
            (batch_size, 0, self.hidden_size), dtype=self.compute_dtype
        )

        has_boxes = False
        if boxes is not None:
            num_boxes = ops.shape(boxes)[1]
            has_boxes = num_boxes > 0

        if points is not None:
            point_embeddings = self._embed_points(
                points, labels, pad=not has_boxes
            )
            sparse_embeddings = ops.concatenate(
                [sparse_embeddings, point_embeddings], axis=1
            )

        if has_boxes:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = ops.concatenate(
                [sparse_embeddings, box_embeddings], axis=1
            )

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            no_mask_vec = self.no_mask_embed(ops.zeros((1,), dtype="int32"))
            no_mask_vec = ops.reshape(no_mask_vec, (1, 1, 1, self.hidden_size))
            dense_embeddings = ops.broadcast_to(
                no_mask_vec,
                (
                    batch_size,
                    self.image_embedding_size[0],
                    self.image_embedding_size[1],
                    self.hidden_size,
                ),
            )

        dense_pe = self.pe_layer.encode_image(self.image_embedding_size)
        dense_pe = dense_pe[None, ...]
        dense_pe = ops.broadcast_to(
            dense_pe,
            (
                batch_size,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
                self.hidden_size,
            ),
        )

        return {
            "prompt_sparse_embeddings": sparse_embeddings,
            "prompt_dense_embeddings": dense_embeddings,
            "prompt_dense_positional_embeddings": dense_pe,
        }

    def compute_output_shape(
        self,
        points_shape=None,
        labels_shape=None,
        boxes_shape=None,
        masks_shape=None,
    ):
        batch_size = None
        if points_shape is not None:
            batch_size = points_shape[0]
        elif boxes_shape is not None:
            batch_size = boxes_shape[0]
        elif masks_shape is not None:
            batch_size = masks_shape[0]

        sparse_dim = 0
        n_boxes = 0
        boxes_known = False

        if boxes_shape is not None:
            if boxes_shape[1] is not None:
                n_boxes = boxes_shape[1]
                boxes_known = True
            else:
                sparse_dim = None
        else:
            boxes_known = True
            n_boxes = 0

        if points_shape is not None:
            if points_shape[1] is not None:
                n_points = points_shape[1]
                if sparse_dim is not None:
                    sparse_dim += n_points
                    if boxes_known:
                        has_boxes = n_boxes > 0
                        if not has_boxes:
                            sparse_dim += 1
            else:
                sparse_dim = None

        if sparse_dim is not None and boxes_known:
            sparse_dim += n_boxes * 2

        return {
            "prompt_sparse_embeddings": (
                batch_size,
                sparse_dim,
                self.hidden_size,
            ),
            "prompt_dense_embeddings": (
                batch_size,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
                self.hidden_size,
            ),
            "prompt_dense_positional_embeddings": (
                batch_size,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
                self.hidden_size,
            ),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "image_embedding_size": self.image_embedding_size,
                "input_image_size": self.input_image_size,
                "mask_in_channels": self.mask_in_channels,
                "activation": self.activation,
            }
        )
        return config
