from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling import Backbone
import torch


class DINOv2(Backbone):
    def __init__(
        self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitl14_reg"
    ):
        super().__init__()
        self.dino_model = torch.hub.load(repo_name, model_name)

        for param in self.dino_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.dino_model.get_intermediate_layers(
            x, return_class_token=False, n=1, reshape=True
        )

    @property
    def size_divisibility(self):
        return 14

    @property
    def padding_constraints(self):
        raise NotImplementedError

    def output_shape(self):
        raise NotImplementedError


@BACKBONE_REGISTRY.register()
def build_dinov2_backbone(cfg, input_shape):
    return DINOv2(
        repo_name=cfg.MODEL.DINO.REPO_NAME, model_name=cfg.MODEL.DINO.MODEL_NAME
    )
