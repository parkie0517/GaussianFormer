from typing import List, Optional
import torch, torch.nn as nn

from mmseg.registry import MODELS
from mmengine import build_from_cfg
from ..base_encoder import BaseEncoder


@MODELS.register_module()
class GaussianOccEncoder(BaseEncoder):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        mid_refine_layer: dict = None,
        spconv_layer: dict = None,
        num_decoder: int = 6,
        operation_order: Optional[List[str]] = None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.anchor_encoder = build(anchor_encoder, MODELS)
        self.op_config_map = {
            "norm": [norm_layer, MODELS],
            "ffn": [ffn, MODELS],
            "deformable": [deformable_model, MODELS],
            "refine": [refine_layer, MODELS],
            "mid_refine":[mid_refine_layer, MODELS],
            "spconv": [spconv_layer, MODELS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        representation,
        rep_features,
        ms_img_feats=None,
        metas=None,
        **kwargs # 나머지 애들은 kwars라는 dict에 저장됨.
    ):
        feature_maps = ms_img_feats # 이게 처음에 extract_img_feature를 사용해서 뽑은 4개 scale의 img feature임
        if isinstance(feature_maps, torch.Tensor): # feature-maps가 torch.tensor인지 확인하는 함수
            feature_maps = [feature_maps] # 아닐 경우, 이렇게 list로 바꿔주기
        instance_feature = rep_features # 쿼리들 (b, 6400, 128)
        anchor = representation # 3DG들 (b, 6400, 28)

        anchor_embed = self.anchor_encoder(anchor) # 3dg의 anchor embedding을 만들어줌. shape = ([b, 6400, 128])

        prediction = []
        for i, op in enumerate(self.operation_order): # 하나의 block 안에 17개의 연산. gformer2는 4개의 block 사용.
            if op == 'spconv':
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor)
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif "refine" in op:
                anchor, gaussian = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                )
            
                prediction.append({'gaussian': gaussian})
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return {"representation": prediction}