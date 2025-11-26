from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.sam2.sam2_backbone import SAM2Backbone


@keras_hub_export("keras_hub.layers.SAM2ImageConverter")
class SAM2ImageConverter(ImageConverter):
    backbone_cls = SAM2Backbone
