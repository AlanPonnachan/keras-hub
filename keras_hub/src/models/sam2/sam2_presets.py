"""SAM2 preset configurations."""

backbone_presets = {
    "sam2_hiera_tiny": {
        "metadata": {
            "description": "SAM2 Tiny model (Hiera-Tiny backbone).",
            "params": 38900000,
            "official_name": "sam2_hiera_t",
            "path": "sam2",
        },
        "kaggle_handle": "kaggle://keras/sam2/keras/sam2_hiera_tiny/1",
    },
    "sam2_hiera_small": {
        "metadata": {
            "description": "SAM2 Small model (Hiera-Small backbone).",
            "params": 46000000,
            "official_name": "sam2_hiera_s",
            "path": "sam2",
        },
        "kaggle_handle": "kaggle://keras/sam2/keras/sam2_hiera_small/1",
    },
    "sam2_hiera_base_plus": {
        "metadata": {
            "description": "SAM2 Base+ model (Hiera-Base+ backbone).",
            "params": 80800000,
            "official_name": "sam2_hiera_b+",
            "path": "sam2",
        },
        "kaggle_handle": "kaggle://keras/sam2/keras/sam2_hiera_base_plus/1",
    },
    "sam2_hiera_large": {
        "metadata": {
            "description": "SAM2 Large model (Hiera-Large backbone).",
            "params": 224400000,
            "official_name": "sam2_hiera_l",
            "path": "sam2",
        },
        "kaggle_handle": "kaggle://keras/sam2/keras/sam2_hiera_large/1",
    },
}
