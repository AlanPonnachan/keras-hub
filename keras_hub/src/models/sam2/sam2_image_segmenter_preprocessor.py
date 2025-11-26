import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)
from keras_hub.src.models.sam2.sam2_backbone import SAM2Backbone
from keras_hub.src.models.sam2.sam2_image_converter import SAM2ImageConverter
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.SAM2ImageSegmenterPreprocessor")
class SAM2ImageSegmenterPreprocessor(ImageSegmenterPreprocessor):
    """SAM2 Image Segmenter Preprocessor.

    This class is responsible for preprocessing inputs for the SAM2 model.
    It applies image resizing and rescaling to the input images using
    `SAM2ImageConverter`, while leaving other inputs (prompts) unchanged.

    Args:
        image_converter: A `keras_hub.layers.SAM2ImageConverter` instance.
            If `None`, a default converter is created.
    """

    backbone_cls = SAM2Backbone
    image_converter_cls = SAM2ImageConverter

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        if "images" in x:
            images = x["images"]
            if self.image_converter:
                x["images"] = self.image_converter(images)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
