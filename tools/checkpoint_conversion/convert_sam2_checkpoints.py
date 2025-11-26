import argparse
import os

import keras
import numpy as np
from sam2.build_sam import build_sam2

from keras_hub.src.models.sam2.sam2_image_segmenter import SAM2ImageSegmenter

PRESET_TO_CONFIG = {
    "sam2_hiera_tiny": "sam2_hiera_t.yaml",
    "sam2_hiera_small": "sam2_hiera_s.yaml",
    "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
    "sam2_hiera_large": "sam2_hiera_l.yaml",
}

PRESET_TO_CHECKPOINT = {
    "sam2_hiera_tiny": "sam2_hiera_tiny.pt",
    "sam2_hiera_small": "sam2_hiera_small.pt",
    "sam2_hiera_base_plus": "sam2_hiera_base_plus.pt",
    "sam2_hiera_large": "sam2_hiera_large.pt",
}


def convert_linear(keras_layer, torch_layer):
    """Copies weights from a PyTorch Linear layer to a Keras Dense layer."""
    w = torch_layer.weight.detach().cpu().numpy().T
    b = torch_layer.bias.detach().cpu().numpy()
    keras_layer.set_weights([w, b])


def convert_conv2d(keras_layer, torch_layer):
    """Copies weights from a PyTorch Conv2d layer to a Keras Conv2D layer."""
    w = torch_layer.weight.detach().cpu().numpy()
    w = np.transpose(w, (2, 3, 1, 0))
    if torch_layer.bias is not None:
        b = torch_layer.bias.detach().cpu().numpy()
        keras_layer.set_weights([w, b])
    else:
        keras_layer.set_weights([w])


def convert_layer_norm(keras_layer, torch_layer):
    """Copies weights from a PyTorch LayerNorm to Keras LayerNormalization."""
    gamma = torch_layer.weight.detach().cpu().numpy()
    beta = torch_layer.bias.detach().cpu().numpy()
    keras_layer.set_weights([gamma, beta])


def convert_embedding(keras_layer, torch_layer):
    """Copies weights from PyTorch Embedding to Keras Embedding."""
    w = torch_layer.weight.detach().cpu().numpy()
    keras_layer.set_weights([w])


def convert_hiera_block(keras_block, torch_block):
    convert_layer_norm(keras_block.norm1, torch_block.norm1)
    convert_layer_norm(keras_block.norm2, torch_block.norm2)

    convert_linear(keras_block.attn.qkv, torch_block.attn.qkv)
    convert_linear(keras_block.attn.proj, torch_block.attn.proj)

    convert_linear(keras_block.mlp.layers[0], torch_block.mlp.layers[0])
    convert_linear(keras_block.mlp.layers[2], torch_block.mlp.layers[2])

    if keras_block.proj is not None:
        convert_linear(keras_block.proj, torch_block.proj)


def convert_image_encoder(keras_img_enc, torch_img_enc):
    print("Converting Image Encoder...")

    convert_conv2d(
        keras_img_enc.trunk.patch_embed.proj,
        torch_img_enc.trunk.patch_embed.proj,
    )

    pos_embed = torch_img_enc.trunk.pos_embed.detach().cpu().numpy()
    pos_embed = np.transpose(pos_embed, (0, 2, 3, 1))
    keras_img_enc.trunk.pos_embed.assign(pos_embed)

    pos_embed_window = (
        torch_img_enc.trunk.pos_embed_window.detach().cpu().numpy()
    )
    pos_embed_window = np.transpose(pos_embed_window, (0, 2, 3, 1))
    keras_img_enc.trunk.pos_embed_window.assign(pos_embed_window)

    for i, block in enumerate(keras_img_enc.trunk.blocks):
        convert_hiera_block(block, torch_img_enc.trunk.blocks[i])

    for i, conv in enumerate(keras_img_enc.neck.convs):
        convert_conv2d(conv, torch_img_enc.neck.convs[i].conv)


def convert_memory_attention(keras_mem_attn, torch_mem_attn):
    print("Converting Memory Attention...")

    for i, layer in enumerate(keras_mem_attn.layers):
        torch_layer = torch_mem_attn.layers[i]

        convert_linear(layer.self_attn.q_proj, torch_layer.self_attn.q_proj)
        convert_linear(layer.self_attn.k_proj, torch_layer.self_attn.k_proj)
        convert_linear(layer.self_attn.v_proj, torch_layer.self_attn.v_proj)
        convert_linear(layer.self_attn.out_proj, torch_layer.self_attn.out_proj)
        convert_layer_norm(layer.norm1, torch_layer.norm1)

        convert_linear(
            layer.cross_attn.q_proj, torch_layer.cross_attn_image.q_proj
        )
        convert_linear(
            layer.cross_attn.k_proj, torch_layer.cross_attn_image.k_proj
        )
        convert_linear(
            layer.cross_attn.v_proj, torch_layer.cross_attn_image.v_proj
        )
        convert_linear(
            layer.cross_attn.out_proj, torch_layer.cross_attn_image.out_proj
        )
        convert_layer_norm(layer.norm2, torch_layer.norm2)

        convert_layer_norm(layer.norm3, torch_layer.norm3)
        convert_linear(layer.linear1, torch_layer.linear1)
        convert_linear(layer.linear2, torch_layer.linear2)

    convert_layer_norm(keras_mem_attn.norm, torch_mem_attn.norm)


def convert_memory_encoder(keras_mem_enc, torch_mem_enc):
    print("Converting Memory Encoder...")

    for i, layer in enumerate(keras_mem_enc.mask_downsampler.encoder.layers):
        torch_l = torch_mem_enc.mask_downsampler.encoder[i]
        if isinstance(layer, keras.layers.Conv2D):
            convert_conv2d(layer, torch_l)
        elif isinstance(layer, keras.layers.LayerNormalization):
            convert_layer_norm(layer, torch_l)

    convert_conv2d(keras_mem_enc.pix_feat_proj, torch_mem_enc.pix_feat_proj)

    for i, layer in enumerate(keras_mem_enc.fuser_layers):
        torch_l = torch_mem_enc.fuser.layers[i]

        w = torch_l.dwconv.weight.detach().cpu().numpy()
        w = np.transpose(w, (2, 3, 0, 1))
        layer.dwconv.set_weights(
            [w, torch_l.dwconv.bias.detach().cpu().numpy()]
        )

        convert_layer_norm(layer.norm, torch_l.norm)
        convert_linear(layer.pwconv1, torch_l.pwconv1)
        convert_linear(layer.pwconv2, torch_l.pwconv2)
        layer.gamma.assign(torch_l.gamma.detach().cpu().numpy())

    if keras_mem_enc.out_proj:
        convert_conv2d(keras_mem_enc.out_proj, torch_mem_enc.out_proj)


def convert_prompt_encoder(keras_prompt, torch_prompt):
    print("Converting Prompt Encoder...")

    keras_prompt.pe_layer.positional_encoding_gaussian_matrix.assign(
        torch_prompt.pe_layer.positional_encoding_gaussian_matrix.detach()
        .cpu()
        .numpy()
    )

    for i in range(4):
        convert_embedding(
            keras_prompt.point_embeddings[i], torch_prompt.point_embeddings[i]
        )

    convert_embedding(
        keras_prompt.not_a_point_embed, torch_prompt.not_a_point_embed
    )
    convert_embedding(keras_prompt.no_mask_embed, torch_prompt.no_mask_embed)

    for i, layer in enumerate(keras_prompt.mask_downscaler.layers):
        torch_l = torch_prompt.mask_downscaling[i]
        if isinstance(layer, keras.layers.Conv2D):
            convert_conv2d(layer, torch_l)
        elif isinstance(layer, keras.layers.LayerNormalization):
            convert_layer_norm(layer, torch_l)


def convert_mask_decoder(keras_decoder, torch_decoder):
    print("Converting Mask Decoder...")

    for i, layer in enumerate(keras_decoder.transformer.layers):
        torch_layer = torch_decoder.transformer.layers[i]

        convert_linear(
            layer.self_attention.q_proj, torch_layer.self_attn.q_proj
        )
        convert_linear(
            layer.self_attention.k_proj, torch_layer.self_attn.k_proj
        )
        convert_linear(
            layer.self_attention.v_proj, torch_layer.self_attn.v_proj
        )
        convert_linear(
            layer.self_attention.out_proj, torch_layer.self_attn.out_proj
        )
        convert_layer_norm(layer.layer_norm1, torch_layer.norm1)

        convert_linear(
            layer.cross_attention_token_to_image.q_proj,
            torch_layer.cross_attn_token_to_image.q_proj,
        )
        convert_linear(
            layer.cross_attention_token_to_image.k_proj,
            torch_layer.cross_attn_token_to_image.k_proj,
        )
        convert_linear(
            layer.cross_attention_token_to_image.v_proj,
            torch_layer.cross_attn_token_to_image.v_proj,
        )
        convert_linear(
            layer.cross_attention_token_to_image.out_proj,
            torch_layer.cross_attn_token_to_image.out_proj,
        )
        convert_layer_norm(layer.layer_norm2, torch_layer.norm2)

        convert_linear(layer.mlp_block.layers[0], torch_layer.mlp.layers[0])
        convert_linear(layer.mlp_block.layers[2], torch_layer.mlp.layers[2])
        convert_layer_norm(layer.layer_norm3, torch_layer.norm3)

        convert_linear(
            layer.cross_attention_image_to_token.q_proj,
            torch_layer.cross_attn_image_to_token.q_proj,
        )
        convert_linear(
            layer.cross_attention_image_to_token.k_proj,
            torch_layer.cross_attn_image_to_token.k_proj,
        )
        convert_linear(
            layer.cross_attention_image_to_token.v_proj,
            torch_layer.cross_attn_image_to_token.v_proj,
        )
        convert_linear(
            layer.cross_attention_image_to_token.out_proj,
            torch_layer.cross_attn_image_to_token.out_proj,
        )
        convert_layer_norm(layer.layer_norm4, torch_layer.norm4)

    convert_linear(
        keras_decoder.transformer.final_attn_token_to_image.q_proj,
        torch_decoder.transformer.final_attn_token_to_image.q_proj,
    )
    convert_linear(
        keras_decoder.transformer.final_attn_token_to_image.k_proj,
        torch_decoder.transformer.final_attn_token_to_image.k_proj,
    )
    convert_linear(
        keras_decoder.transformer.final_attn_token_to_image.v_proj,
        torch_decoder.transformer.final_attn_token_to_image.v_proj,
    )
    convert_linear(
        keras_decoder.transformer.final_attn_token_to_image.out_proj,
        torch_decoder.transformer.final_attn_token_to_image.out_proj,
    )
    convert_layer_norm(
        keras_decoder.transformer.norm_final_attn,
        torch_decoder.transformer.norm_final_attn,
    )

    convert_embedding(keras_decoder.iou_token, torch_decoder.iou_token)
    convert_embedding(keras_decoder.mask_tokens, torch_decoder.mask_tokens)
    if keras_decoder.pred_obj_scores:
        convert_embedding(
            keras_decoder.obj_score_token, torch_decoder.obj_score_token
        )
        convert_linear(
            keras_decoder.pred_obj_score_head, torch_decoder.pred_obj_score_head
        )

    w = torch_decoder.output_upscaling[0].weight.detach().cpu().numpy()
    w = np.transpose(w, (2, 3, 1, 0))
    b = torch_decoder.output_upscaling[0].bias.detach().cpu().numpy()
    keras_decoder.upscale_conv1.set_weights([w, b])

    convert_layer_norm(
        keras_decoder.upscale_layer_norm, torch_decoder.output_upscaling[1]
    )

    w = torch_decoder.output_upscaling[3].weight.detach().cpu().numpy()
    w = np.transpose(w, (2, 3, 1, 0))
    b = torch_decoder.output_upscaling[3].bias.detach().cpu().numpy()
    keras_decoder.upscale_conv2.set_weights([w, b])

    for i in range(keras_decoder.num_mask_tokens):
        k_mlp = keras_decoder.output_hypernetworks_mlps[i]
        t_mlp = torch_decoder.output_hypernetworks_mlps[i]
        convert_linear(k_mlp.layers[0], t_mlp.layers[0])
        convert_linear(k_mlp.layers[2], t_mlp.layers[2])
        convert_linear(k_mlp.layers[4], t_mlp.layers[4])

    convert_linear(
        keras_decoder.iou_prediction_head.layers[0],
        torch_decoder.iou_prediction_head.layers[0],
    )
    convert_linear(
        keras_decoder.iou_prediction_head.layers[2],
        torch_decoder.iou_prediction_head.layers[2],
    )
    convert_linear(
        keras_decoder.iou_prediction_head.layers[4],
        torch_decoder.iou_prediction_head.layers[4],
    )


def main(preset):
    print(f"Converting preset: {preset}")

    segmenter = SAM2ImageSegmenter.from_preset(preset, load_weights=False)
    backbone = segmenter.backbone

    cfg_name = PRESET_TO_CONFIG[preset]
    ckpt_name = PRESET_TO_CHECKPOINT[preset]

    if not os.path.exists(ckpt_name):
        print(f"Downloading {ckpt_name}...")
        os.system(
            f"wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/{ckpt_name}"
        )

    print(f"Loading PyTorch model from {cfg_name}...")

    torch_model = build_sam2(cfg_name, ckpt_name)
    torch_model.eval()

    convert_image_encoder(backbone.image_encoder, torch_model.image_encoder)
    convert_memory_attention(
        backbone.memory_attention, torch_model.memory_attention
    )
    convert_memory_encoder(backbone.memory_encoder, torch_model.memory_encoder)
    convert_prompt_encoder(
        backbone.prompt_encoder, torch_model.sam_prompt_encoder
    )
    convert_mask_decoder(backbone.mask_decoder, torch_model.sam_mask_decoder)

    backbone.no_mem_layer.no_mem_embed.assign(
        torch_model.no_mem_embed.detach().cpu().numpy()
    )
    backbone.no_mem_layer.no_mem_pos_enc.assign(
        torch_model.no_mem_pos_enc.detach().cpu().numpy()
    )

    print(f"Saving to {preset}...")
    segmenter.save_to_preset(f"./{preset}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, required=True)
    args = parser.parse_args()
    main(args.preset)
