from . import data  # register all new datasets
from . import modeling

# config
from .config import add_model_2d_config, add_depr_config

# dataset loading
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

from .data.dataset_mappers.front3d_pifu_dataset_mapper import Front3DPIFuDatasetMapper

# models
from .depr_model import DepR

