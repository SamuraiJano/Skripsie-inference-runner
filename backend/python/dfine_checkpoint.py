"""Load official D-FINE nano training checkpoints into Transformers.

The key conversion is adapted from Hugging Face Transformers' Apache-2.0
D-FINE conversion utility.  Keeping it local makes model activation offline and
avoids importing executable objects from a training checkpoint.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_KEY_MAPPING = {
    r"decoder.up": r"model.decoder.up",
    r"decoder.reg_scale": r"model.decoder.reg_scale",
    r"backbone.stem.stem1.conv.weight": r"model.backbone.model.embedder.stem1.convolution.weight",
    r"backbone.stem.stem2a.conv.weight": r"model.backbone.model.embedder.stem2a.convolution.weight",
    r"backbone.stem.stem2b.conv.weight": r"model.backbone.model.embedder.stem2b.convolution.weight",
    r"backbone.stem.stem3.conv.weight": r"model.backbone.model.embedder.stem3.convolution.weight",
    r"backbone.stem.stem4.conv.weight": r"model.backbone.model.embedder.stem4.convolution.weight",
    r"backbone.stem.(stem1|stem2a|stem2b|stem3|stem4).bn.(weight|bias|running_mean|running_var)":
        r"model.backbone.model.embedder.\1.normalization.\2",
    r"backbone.stem.(stem1|stem2a|stem2b|stem3|stem4).lab.(scale|bias)":
        r"model.backbone.model.embedder.\1.lab.\2",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv.weight":
        r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).bn.(weight|bias|running_mean|running_var)":
        r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.normalization.\4",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv(1|2).conv.weight":
        r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv\4.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv(1|2).bn.(weight|bias|running_mean|running_var)":
        r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv\4.normalization.\5",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.(0|1).conv.weight":
        r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.\3.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.(0|1).bn.(weight|bias|running_mean|running_var)":
        r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.\3.normalization.\4",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.(0|1).lab.(scale|bias)":
        r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.\3.lab.\4",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).lab.(scale|bias)":
        r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.lab.\4",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv(1|2).lab.(scale|bias)":
        r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv\4.lab.\5",
    r"backbone.stages.(\d+).downsample.conv.weight":
        r"model.backbone.model.encoder.stages.\1.downsample.convolution.weight",
    r"backbone.stages.(\d+).downsample.bn.(weight|bias|running_mean|running_var)":
        r"model.backbone.model.encoder.stages.\1.downsample.normalization.\2",
    r"backbone.stages.(\d+).downsample.lab.(scale|bias)":
        r"model.backbone.model.encoder.stages.\1.downsample.lab.\2",
    r"encoder.encoder.(\d+).layers.0.self_attn.out_proj.(weight|bias)":
        r"model.encoder.encoder.\1.layers.0.self_attn.out_proj.\2",
    r"encoder.encoder.(\d+).layers.0.linear1.(weight|bias)": r"model.encoder.encoder.\1.layers.0.fc1.\2",
    r"encoder.encoder.(\d+).layers.0.linear2.(weight|bias)": r"model.encoder.encoder.\1.layers.0.fc2.\2",
    r"encoder.encoder.(\d+).layers.0.norm1.(weight|bias)":
        r"model.encoder.encoder.\1.layers.0.self_attn_layer_norm.\2",
    r"encoder.encoder.(\d+).layers.0.norm2.(weight|bias)":
        r"model.encoder.encoder.\1.layers.0.final_layer_norm.\2",
    r"encoder.input_proj.(\d+).conv.weight": r"model.encoder_input_proj.\1.0.weight",
    r"encoder.input_proj.(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder_input_proj.\1.1.\2",
    r"encoder.lateral_convs.(\d+).conv.weight": r"model.encoder.lateral_convs.\1.conv.weight",
    r"encoder.lateral_convs.(\d+).norm.(weight|bias|running_mean|running_var)":
        r"model.encoder.lateral_convs.\1.norm.\2",
    r"encoder.(fpn|pan)_blocks.(\d+).cv1.conv.weight": r"model.encoder.\1_blocks.\2.conv1.conv.weight",
    r"encoder.(fpn|pan)_blocks.(\d+).cv1.norm.(weight|bias|running_mean|running_var)":
        r"model.encoder.\1_blocks.\2.conv1.norm.\3",
    r"encoder.(fpn|pan)_blocks.(\d+).cv2.0.conv(1|2).conv.weight":
        r"model.encoder.\1_blocks.\2.csp_rep1.conv\3.conv.weight",
    r"encoder.(fpn|pan)_blocks.(\d+).cv2.0.conv(1|2).norm.(weight|bias|running_mean|running_var)":
        r"model.encoder.\1_blocks.\2.csp_rep1.conv\3.norm.\4",
    r"encoder.(fpn|pan)_blocks.(\d+).cv2.1.conv.weight": r"model.encoder.\1_blocks.\2.conv2.conv.weight",
    r"encoder.(fpn|pan)_blocks.(\d+).cv2.1.norm.(weight|bias|running_mean|running_var)":
        r"model.encoder.\1_blocks.\2.conv2.norm.\3",
    r"encoder.(fpn|pan)_blocks.(\d+).cv3.0.conv(1|2).conv.weight":
        r"model.encoder.\1_blocks.\2.csp_rep2.conv\3.conv.weight",
    r"encoder.(fpn|pan)_blocks.(\d+).cv3.0.conv(1|2).norm.(weight|bias|running_mean|running_var)":
        r"model.encoder.\1_blocks.\2.csp_rep2.conv\3.norm.\4",
    r"encoder.(fpn|pan)_blocks.(\d+).cv3.1.conv.weight": r"model.encoder.\1_blocks.\2.conv3.conv.weight",
    r"encoder.(fpn|pan)_blocks.(\d+).cv3.1.norm.(weight|bias|running_mean|running_var)":
        r"model.encoder.\1_blocks.\2.conv3.norm.\3",
    r"encoder.(fpn|pan)_blocks.(\d+).cv4.conv.weight": r"model.encoder.\1_blocks.\2.conv4.conv.weight",
    r"encoder.(fpn|pan)_blocks.(\d+).cv4.norm.(weight|bias|running_mean|running_var)":
        r"model.encoder.\1_blocks.\2.conv4.norm.\3",
    r"encoder.(fpn|pan)_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv(1|2).conv.weight":
        r"model.encoder.\1_blocks.\2.csp_rep1.bottlenecks.\3.conv\4.conv.weight",
    r"encoder.(fpn|pan)_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv(1|2).norm.(weight|bias|running_mean|running_var)":
        r"model.encoder.\1_blocks.\2.csp_rep1.bottlenecks.\3.conv\4.norm.\5",
    r"encoder.(fpn|pan)_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv(1|2).conv.weight":
        r"model.encoder.\1_blocks.\2.csp_rep2.bottlenecks.\3.conv\4.conv.weight",
    r"encoder.(fpn|pan)_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv(1|2).norm.(weight|bias|running_mean|running_var)":
        r"model.encoder.\1_blocks.\2.csp_rep2.bottlenecks.\3.conv\4.norm.\5",
    r"encoder.downsample_convs.(\d+).0.cv(\d+).conv.weight": r"model.encoder.downsample_convs.\1.conv\2.conv.weight",
    r"encoder.downsample_convs.(\d+).0.cv(\d+).norm.(weight|bias|running_mean|running_var)":
        r"model.encoder.downsample_convs.\1.conv\2.norm.\3",
    r"decoder.decoder.layers.(\d+).self_attn.out_proj.(weight|bias)":
        r"model.decoder.layers.\1.self_attn.out_proj.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.sampling_offsets.(weight|bias)":
        r"model.decoder.layers.\1.encoder_attn.sampling_offsets.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.attention_weights.(weight|bias)":
        r"model.decoder.layers.\1.encoder_attn.attention_weights.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.value_proj.(weight|bias)":
        r"model.decoder.layers.\1.encoder_attn.value_proj.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.output_proj.(weight|bias)":
        r"model.decoder.layers.\1.encoder_attn.output_proj.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.num_points_scale":
        r"model.decoder.layers.\1.encoder_attn.num_points_scale",
    r"decoder.decoder.layers.(\d+).gateway.gate.(weight|bias)": r"model.decoder.layers.\1.gateway.gate.\2",
    r"decoder.decoder.layers.(\d+).gateway.norm.(weight|bias)": r"model.decoder.layers.\1.gateway.norm.\2",
    r"decoder.decoder.layers.(\d+).norm1.(weight|bias)":
        r"model.decoder.layers.\1.self_attn_layer_norm.\2",
    r"decoder.decoder.layers.(\d+).norm2.(weight|bias)":
        r"model.decoder.layers.\1.encoder_attn_layer_norm.\2",
    r"decoder.decoder.layers.(\d+).norm3.(weight|bias)": r"model.decoder.layers.\1.final_layer_norm.\2",
    r"decoder.decoder.layers.(\d+).linear1.(weight|bias)": r"model.decoder.layers.\1.fc1.\2",
    r"decoder.decoder.layers.(\d+).linear2.(weight|bias)": r"model.decoder.layers.\1.fc2.\2",
    r"decoder.decoder.lqe_layers.(\d+).reg_conf.layers.(\d+).(weight|bias)":
        r"model.decoder.lqe_layers.\1.reg_conf.layers.\2.\3",
    r"decoder.dec_score_head.(\d+).(weight|bias)": r"model.decoder.class_embed.\1.\2",
    r"decoder.dec_bbox_head.(\d+).layers.(\d+).(weight|bias)":
        r"model.decoder.bbox_embed.\1.layers.\2.\3",
    r"decoder.pre_bbox_head.layers.(\d+).(weight|bias)": r"model.decoder.pre_bbox_head.layers.\1.\2",
    r"decoder.input_proj.(\d+).conv.weight": r"model.decoder_input_proj.\1.0.weight",
    r"decoder.input_proj.(\d+).norm.(weight|bias|running_mean|running_var)": r"model.decoder_input_proj.\1.1.\2",
    r"decoder.denoising_class_embed.weight": r"model.denoising_class_embed.weight",
    r"decoder.query_pos_head.layers.(\d+).(weight|bias)": r"model.decoder.query_pos_head.layers.\1.\2",
    r"decoder.enc_output.proj.(weight|bias)": r"model.enc_output.0.\1",
    r"decoder.enc_output.norm.(weight|bias)": r"model.enc_output.1.\1",
    r"decoder.enc_score_head.(weight|bias)": r"model.enc_score_head.\1",
    r"decoder.enc_bbox_head.layers.(\d+).(weight|bias)": r"model.enc_bbox_head.layers.\1.\2",
}


def checkpoint_state(path: Path) -> dict[str, Any]:
    """Read tensor-only state from an official D-FINE training checkpoint."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("D-FINE checkpoint must contain a state dictionary")
    ema = checkpoint.get("ema")
    state = ema.get("module") if isinstance(ema, dict) else None
    if not isinstance(state, dict):
        state = checkpoint.get("model")
    if not isinstance(state, dict):
        state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("D-FINE checkpoint has no model or EMA tensor state")
    required = ("backbone.stem.stem1.conv.weight", "decoder.enc_score_head.weight")
    if any(key not in state for key in required):
        raise ValueError(".pth file is not an official D-FINE checkpoint")
    return dict(state)


def checkpoint_labels(state: dict[str, Any]) -> list[str]:
    score_head = state["decoder.enc_score_head.weight"]
    count = int(score_head.shape[0])
    return ["object"] if count == 1 else [f"class_{index}" for index in range(count)]


def _nano_config(labels: list[str]) -> Any:
    from transformers import DFineConfig

    config = DFineConfig()
    config.num_labels = len(labels)
    config.id2label = {index: label for index, label in enumerate(labels)}
    config.label2id = {label: index for index, label in enumerate(labels)}
    backbone = config.backbone_config
    backbone.hidden_sizes = [128, 256, 512, 1024]
    backbone.layer_type = "basic"
    backbone.embedding_size = 32
    backbone.stem_channels = [3, 16, 16]
    backbone.stage_in_channels = [16, 64, 256, 512]
    backbone.stage_mid_channels = [16, 32, 64, 128]
    backbone.stage_out_channels = [64, 256, 512, 1024]
    backbone.stage_num_blocks = [1, 1, 2, 1]
    backbone.stage_downsample = [False, True, True, True]
    backbone.stage_light_block = [False, False, True, True]
    backbone.stage_kernel_size = [3, 3, 5, 5]
    backbone.stage_numb_of_layers = [3, 3, 3, 3]
    backbone.out_indices = [3, 4]
    backbone.use_learnable_affine_block = True
    config.num_feature_levels = 2
    config.encoder_ffn_dim = 512
    config.encode_proj_layers = [1]
    config.d_model = 128
    config.encoder_hidden_dim = 128
    config.decoder_ffn_dim = 512
    config.encoder_in_channels = [512, 1024]
    config.decoder_n_points = [6, 6]
    config.decoder_in_channels = [128, 128]
    config.feat_strides = [16, 32]
    config.depth_mult = 0.5
    config.decoder_layers = 3
    config.hidden_expansion = 0.34
    return config


def _is_nano(state: dict[str, Any]) -> bool:
    score_head = state["decoder.enc_score_head.weight"]
    decoder_layers = {
        int(match.group(1))
        for key in state
        if (match := re.match(r"decoder\.decoder\.layers\.(\d+)\.", key))
    }
    return int(score_head.shape[1]) == 128 and decoder_layers == {0, 1, 2}


def validate_dfine_nano(state: dict[str, Any]) -> None:
    if not _is_nano(state):
        raise ValueError("only D-FINE nano .pth checkpoints are currently supported")


def _convert_state(state: dict[str, Any], config: Any) -> dict[str, Any]:
    """Translate the official training checkpoint layout to Transformers.

    This changes parameter *names and layout*, not learned values.  Strict
    loading at the end is intentional: a silently missing parameter would run
    a partially initialized detector and produce plausible-looking bad output.
    """
    import torch

    validate_dfine_nano(state)
    converted = dict(state)
    converted.pop("decoder.valid_mask", None)
    converted.pop("decoder.anchors", None)

    for original, replacement in _KEY_MAPPING.items():
        for key in list(converted):
            new_key = re.sub(original, replacement, key)
            if new_key != key:
                converted[new_key] = converted.pop(key)

    converted.pop("decoder.model.decoder.up", None)
    converted.pop("decoder.model.decoder.reg_scale", None)

    hidden = int(config.encoder_hidden_dim)
    for index in range(int(config.encoder_layers)):
        # PyTorch MultiheadAttention stores Q/K/V as one fused matrix.  The
        # Transformers D-FINE implementation expects three projection modules.
        weight = converted.pop(f"encoder.encoder.{index}.layers.0.self_attn.in_proj_weight")
        bias = converted.pop(f"encoder.encoder.{index}.layers.0.self_attn.in_proj_bias")
        prefix = f"model.encoder.encoder.{index}.layers.0.self_attn"
        converted[f"{prefix}.q_proj.weight"] = weight[:hidden]
        converted[f"{prefix}.q_proj.bias"] = bias[:hidden]
        converted[f"{prefix}.k_proj.weight"] = weight[hidden : 2 * hidden]
        converted[f"{prefix}.k_proj.bias"] = bias[hidden : 2 * hidden]
        converted[f"{prefix}.v_proj.weight"] = weight[-hidden:]
        converted[f"{prefix}.v_proj.bias"] = bias[-hidden:]

    for index in range(int(config.decoder_layers)):
        weight = converted.pop(f"decoder.decoder.layers.{index}.self_attn.in_proj_weight")
        bias = converted.pop(f"decoder.decoder.layers.{index}.self_attn.in_proj_bias")
        prefix = f"model.decoder.layers.{index}.self_attn"
        converted[f"{prefix}.q_proj.weight"] = weight[:128]
        converted[f"{prefix}.q_proj.bias"] = bias[:128]
        converted[f"{prefix}.k_proj.weight"] = weight[256:384]
        converted[f"{prefix}.k_proj.bias"] = bias[256:384]
        converted[f"{prefix}.v_proj.weight"] = weight[-128:]
        converted[f"{prefix}.v_proj.bias"] = bias[-128:]

    for key in list(converted):
        if key.endswith("num_batches_tracked"):
            del converted[key]
        elif "bbox_embed" in key or ("class_embed" in key and "denoising_" not in key):
            # Transformers registers the shared decoder heads at an additional
            # path, so expose the same tensors under both expected names.
            converted[key.split("model.decoder.")[-1]] = converted[key]

    if not all(isinstance(value, torch.Tensor) for value in converted.values()):
        raise ValueError("D-FINE state contains non-tensor model values")
    return converted


def load_dfine_nano(path: Path, device: str) -> tuple[Any, Any, list[str]]:
    """Build and strictly load a D-FINE nano model without network access."""
    from transformers import DFineForObjectDetection, RTDetrImageProcessor

    state = checkpoint_state(path)
    labels = checkpoint_labels(state)
    config = _nano_config(labels)
    model = DFineForObjectDetection(config)
    model.load_state_dict(_convert_state(state, config), strict=True)
    model.to(device).eval()
    processor = RTDetrImageProcessor(size={"height": 640, "width": 640})
    return processor, model, labels
