# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from typing import Any, Dict, Optional

from transformers import PretrainedConfig

class TouchASUConfig(PretrainedConfig):
    model_type = "touch_asu"

    def __init__(
        self,
        llm_model_name_or_path: str = 'Qwen/Qwen2-7B',
        wenet_model_name_or_path: str = '',
        pretrained_ckpt_path: Optional[str] = None,
        encoder_ds_rate: int = 4,
        encoder_projector_ds_rate: int = 2,
        projector_hidden_size: int = 2048,
        hidden_size: int = 0,  # Will override in TouchASU Model
        lora_config: Optional[Dict[str, Any]] = None,
        # max_speech_frames: int = 2000,  # 20s
        max_speech_frames: int = 15000,  # 150s
        min_speech_frames: int = 20,  # 0.2s
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_model_name_or_path = llm_model_name_or_path
        self.wenet_model_name_or_path = wenet_model_name_or_path
        self.encoder_ds_rate = encoder_ds_rate
        self.encoder_projector_ds_rate = encoder_projector_ds_rate
        self.projector_hidden_size = projector_hidden_size
        self.lora_config = lora_config
        self.hidden_size = hidden_size
        self.max_speech_frames = max_speech_frames
        self.min_speech_frames = min_speech_frames
        self.pretrained_ckpt_path = pretrained_ckpt_path

class TouchASUWithDiarModuleConfig(TouchASUConfig):
    """ Config for TouchASU with Diarization Module
    """
    model_type = "touch_asu_with_diar_module"
    
    def __init__(
        self,
        diar_model_path: str = '/lihaoyu/workspace/SA-ASR/StreamingSortformer/pretrained/diar_streaming_sortformer_4spk-v2.nemo',
        diar_emb_size: int = 4,             # 注意这里是不做time pooling之前的diar embedding维度
        diar_downsample_rate: int = 5,      # 目前不涉及对齐问题，因为是T维度的拼接而不是特征维度的concat
        asr_projector_frozen: bool = False, # 是否冻结ASR的projector
        diar_projector_frozen: bool = False,# 是否冻结diar的projector
        diar_module_frozen: bool = True,    # 是否冻结diar的downsampler
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diar_model_name_or_path = diar_model_path
        self.diar_emb_size = diar_emb_size
        self.diar_downsample_rate = diar_downsample_rate

        self.asr_projector_frozen = asr_projector_frozen
        self.diar_projector_frozen = diar_projector_frozen
        self.diar_module_frozen = diar_module_frozen

__all__ = ["TouchASUConfig", "TouchASUWithDiarModuleConfig"]
