# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from typing import Any, Dict, Optional

from transformers import PretrainedConfig

# for speaker attributed ASU
import sys
import traceback
sys.path.append('/wangshuai/workspace/west/examples/aishell/asr/3D-Speaker')
try:
    from speakerlab.bin.infer_sv_batch import supports as SV_SUPPORTS
except ImportError:
    traceback.print_exc()
    raise ImportError("Failed to import supports from speakerlab.bin.infer_sv_batch. "
                      "Please ensure that the 3D-Speaker repository is correctly cloned and accessible.")

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
        max_speech_frames: int = 2000,  # 20s
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

class TouchASUSpeakerAttributedConfig(TouchASUConfig):
    """ Config for TouchASU with Speaker Encoder
    """
    model_type = "touch_asu_speaker_attributed"
    
    def __init__(
        self,
        speaker_encoder_id: str = 'iic/speech_campplus_sv_zh_en_16k-common_advanced',
        speaker_emb_size: int = 512,        # 注意这里是不做time pooling之前的speaker embedding维度
        speaker_downsample_rate: int = 4,   # 需要手动计算一下，FireRedASR是下采样8倍，Speaker Encoder是下采样2倍，所以这里是4倍
        asr_projector_frozen: bool = True,  # 是否冻结ASR的projector
        speaker_projector_frozen: bool = False,# 是否冻结speaker的projector
        speaker_downsampler_frozen: bool = False, # 是否冻结speaker的downsampler
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.speaker_encoder_id = speaker_encoder_id
        self.speaker_emb_size = speaker_emb_size
        self.speaker_downsample_rate = speaker_downsample_rate
        assert self.speaker_encoder_id in SV_SUPPORTS, \
            f"Speaker encoder {self.speaker_encoder_id} is not supported. " \
            f"Supported models are: {SV_SUPPORTS.keys()}"

        self.asr_projector_frozen = asr_projector_frozen
        self.speaker_projector_frozen = speaker_projector_frozen
        self.speaker_downsampler_frozen = speaker_downsampler_frozen

__all__ = ["TouchASUConfig", "TouchASUSpeakerAttributedConfig"]
