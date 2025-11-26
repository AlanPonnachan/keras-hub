import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.sam2.sam2_layers import MLP
from keras_hub.src.models.sam2.sam2_transformer import TwoWayTransformer


@keras_hub_export("keras_hub.layers.SAM2MaskDecoder")
class SAM2MaskDecoder(keras.layers.Layer):
    """Mask decoder for the Segment Anything Model 2 (SAM2).

    This module maps the image embedding and a set of prompt embeddings to an
    output mask.

    Args:
        hidden_size: int. The hidden size of the TwoWayTransformer.
        num_layers: int. The number of layers in the TwoWayTransformer.
        intermediate_dim: int. The intermediate dimension of the
            TwoWayTransformer.
        num_heads: int. The number of heads in the TwoWayTransformer.
        embedding_dim: int. The number of input features to the
            transformer decoder.
        num_multimask_outputs: int. Number of multimask outputs.
            Defaults to `3`.
        iou_head_depth: int. The depth of the dense net used to
            predict the IoU confidence score. Defaults to `3`.
        iou_head_hidden_dim: int. The number of units in the hidden
            layers used in the dense net to predict the IoU confidence score.
            Defaults to `256`.
        activation: str. Activation to use in the mask upscaler
            network. Defaults to `"gelu"`.
        use_high_res_features: bool. Whether to use high-resolution features
            (skip connections) during upscaling. Defaults to `False`.
        iou_prediction_use_sigmoid: bool. Whether to use sigmoid
        on IoU predictions.
            Defaults to `False`.
        pred_obj_scores: bool. Whether to predict object presence scores.
            Defaults to `False`.
    """

    def __init__(
        self,
        hidden_size,
        num_layers,
        intermediate_dim,
        num_heads,
        embedding_dim,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        activation="gelu",
        use_high_res_features=False,
        iou_prediction_use_sigmoid=False,
        pred_obj_scores=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.activation = activation
        self.use_high_res_features = use_high_res_features
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.pred_obj_scores = pred_obj_scores

        self.transformer = TwoWayTransformer(
            depth=num_layers,
            embedding_dim=hidden_size,
            num_heads=num_heads,
            mlp_dim=intermediate_dim,
            activation="relu",
            attention_downsample_rate=2,
            dtype=self.dtype_policy,
        )

        self.iou_token = keras.layers.Embedding(
            1, embedding_dim, dtype=self.dtype_policy
        )

        if pred_obj_scores:
            self.obj_score_token = keras.layers.Embedding(
                1, embedding_dim, dtype=self.dtype_policy
            )
            self.pred_obj_score_head = keras.layers.Dense(
                1, dtype=self.dtype_policy
            )

        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = keras.layers.Embedding(
            self.num_mask_tokens, embedding_dim, dtype=self.dtype_policy
        )

        self.upscale_conv1 = keras.layers.Conv2DTranspose(
            embedding_dim // 4,
            kernel_size=2,
            strides=2,
            dtype=self.dtype_policy,
        )
        self.upscale_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, dtype=self.dtype_policy
        )
        self.upscale_activation1 = keras.layers.Activation(
            activation, dtype=self.dtype_policy
        )
        self.upscale_conv2 = keras.layers.Conv2DTranspose(
            embedding_dim // 8,
            kernel_size=2,
            strides=2,
            dtype=self.dtype_policy,
        )
        self.upscale_activation2 = keras.layers.Activation(
            activation, dtype=self.dtype_policy
        )

        if use_high_res_features:
            self.conv_s0 = keras.layers.Conv2D(
                embedding_dim // 8, kernel_size=1, dtype=self.dtype_policy
            )
            self.conv_s1 = keras.layers.Conv2D(
                embedding_dim // 4, kernel_size=1, dtype=self.dtype_policy
            )

        self.output_hypernetworks_mlps = [
            MLP(
                hidden_dim=embedding_dim,
                output_dim=embedding_dim // 8,
                num_layers=3,
                activation="relu",
                dtype=self.dtype_policy,
            )
            for _ in range(self.num_mask_tokens)
        ]

        self.iou_prediction_head = MLP(
            hidden_dim=iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,
            num_layers=iou_head_depth,
            activation="relu",
            sigmoid_output=iou_prediction_use_sigmoid,
            dtype=self.dtype_policy,
        )

    def build(self, input_shape=None, **kwargs):
        self.transformer.build()
        self.iou_token.build(None)
        self.mask_tokens.build(None)

        if self.pred_obj_scores:
            self.obj_score_token.build(None)
            self.pred_obj_score_head.build((None, self.embedding_dim))

        upscale_in_shape = (None, None, None, self.embedding_dim)
        self.upscale_conv1.build(upscale_in_shape)

        ln_shape = (None, None, None, self.embedding_dim // 4)
        self.upscale_layer_norm.build(ln_shape)
        self.upscale_activation1.build(ln_shape)

        self.upscale_conv2.build(ln_shape)

        act2_shape = (None, None, None, self.embedding_dim // 8)
        self.upscale_activation2.build(act2_shape)

        for mlp in self.output_hypernetworks_mlps:
            mlp.build((None, self.embedding_dim))

        self.iou_prediction_head.build((None, self.embedding_dim))
        self.built = True

    def call(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output=True,
        high_res_features=None,
        repeat_image=False,
    ):
        tokens_list = []
        if self.pred_obj_scores:
            tokens_list.append(
                self.obj_score_token(ops.zeros((1,), dtype="int32"))
            )

        tokens_list.append(self.iou_token(ops.zeros((1,), dtype="int32")))

        mask_tokens_indices = ops.arange(self.num_mask_tokens, dtype="int32")
        tokens_list.append(self.mask_tokens(mask_tokens_indices))

        output_tokens = ops.concatenate(tokens_list, axis=0)

        batch_size = ops.shape(sparse_prompt_embeddings)[0]
        output_tokens = ops.broadcast_to(
            output_tokens[None, ...],
            (batch_size, ops.shape(output_tokens)[0], self.embedding_dim),
        )

        tokens = ops.concatenate(
            [output_tokens, sparse_prompt_embeddings], axis=1
        )

        if repeat_image:
            src = ops.repeat(image_embeddings, batch_size, axis=0)
            src = src + dense_prompt_embeddings
            pos_src = ops.repeat(image_pe, batch_size, axis=0)
        else:
            src = image_embeddings + dense_prompt_embeddings
            pos_src = ops.broadcast_to(image_pe, ops.shape(src))

        hs, updated_keys = self.transformer(src, pos_src, tokens)

        if self.pred_obj_scores:
            obj_score_token_out = hs[:, 0, :]
            iou_token_out = hs[:, 1, :]
            mask_tokens_out = hs[:, 2 : (2 + self.num_mask_tokens), :]
        else:
            iou_token_out = hs[:, 0, :]
            mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        shape_src = ops.shape(image_embeddings)
        h, w = shape_src[1], shape_src[2]
        updated_src = ops.reshape(
            updated_keys, (batch_size, h, w, self.embedding_dim)
        )

        x = self.upscale_conv1(updated_src)
        if self.use_high_res_features and high_res_features is not None:
            feat_s0, feat_s1 = high_res_features
            feat_s1_proj = self.conv_s1(feat_s1)
            x = x + feat_s1_proj

        x = self.upscale_layer_norm(x)
        x = self.upscale_activation1(x)

        x = self.upscale_conv2(x)
        if self.use_high_res_features and high_res_features is not None:
            feat_s0, feat_s1 = high_res_features
            feat_s0_proj = self.conv_s0(feat_s0)
            x = x + feat_s0_proj

        upscaled_embedding = self.upscale_activation2(x)

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            token_i = mask_tokens_out[:, i, :]
            hyper_in_list.append(self.output_hypernetworks_mlps[i](token_i))

        hyper_in = ops.stack(hyper_in_list, axis=1)

        b, h_up, w_up, c_up = (
            ops.shape(upscaled_embedding)[0],
            ops.shape(upscaled_embedding)[1],
            ops.shape(upscaled_embedding)[2],
            ops.shape(upscaled_embedding)[3],
        )

        upscaled_flat = ops.reshape(upscaled_embedding, (b, h_up * w_up, c_up))
        upscaled_flat = ops.transpose(upscaled_flat, (0, 2, 1))

        masks = ops.matmul(hyper_in, upscaled_flat)
        masks = ops.reshape(masks, (b, self.num_mask_tokens, h_up, w_up))

        iou_pred = self.iou_prediction_head(iou_token_out)

        if self.pred_obj_scores:
            object_score_logits = self.pred_obj_score_head(obj_score_token_out)
        else:
            object_score_logits = 10.0 * ops.ones(
                (batch_size, 1), dtype=self.compute_dtype
            )

        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "sam_tokens": mask_tokens_out,
            "object_score_logits": object_score_logits,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "embedding_dim": self.embedding_dim,
                "num_multimask_outputs": self.num_multimask_outputs,
                "iou_head_depth": self.iou_head_depth,
                "iou_head_hidden_dim": self.iou_head_hidden_dim,
                "activation": self.activation,
                "use_high_res_features": self.use_high_res_features,
                "iou_prediction_use_sigmoid": self.iou_prediction_use_sigmoid,
                "pred_obj_scores": self.pred_obj_scores,
            }
        )
        return config
