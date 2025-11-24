# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from typing import Optional, Dict, List

import torch
import wenet
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          GenerationMixin, PreTrainedModel)

from west.utils.utils import freeze_module

from .configuration_touch_asu import TouchASUConfig
from pathlib import Path
import re
from wenet.dataset.kaldi_io import read_mat

# for speaker attributed ASU
import os
import sys
import traceback
sys.path.append('/wangshuai/workspace/west/examples/aishell/asr/3D-Speaker')
try:
    from speakerlab.process.processor import FBank
    from speakerlab.bin.infer_sv_batch import supports as SV_SUPPORTS
    from speakerlab.utils.builder import dynamic_import
except ImportError:
    traceback.print_exc()
    raise ImportError("Failed to import modules from speakerlab. "
                      "Please ensure that the 3D-Speaker repository is correctly cloned and accessible.")

from modelscope import snapshot_download
from .configuration_touch_asu import TouchASUSpeakerAttributedConfig
import torch.nn.functional as F

from transformers.utils import logging
logger = logging.get_logger(__name__)

class ProjectorCov1d(nn.Module):

    def __init__(self, config, encoder_dim, llm_dim):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.conv1d = nn.Conv1d(in_channels=encoder_dim,
                                out_channels=encoder_dim,
                                kernel_size=self.k,
                                stride=self.k,
                                padding=0)
        self.linear1 = nn.Linear(encoder_dim, config.projector_hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(config.projector_hidden_size, llm_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x

class ProjectorConcat(nn.Module):
    # Simple concatenation + linear projection for Fireredasr
    def __init__(self, encoder_dim, llm_dim, downsample_rate=2):
        super().__init__()
        self.k = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.k, feat_dim * self.k
        )

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

class TouchASU(PreTrainedModel, GenerationMixin):
    """ LLM based Automatic Speech Understanding
    """
    model_type = 'touch_asu'
    config_class = TouchASUConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: TouchASUConfig):
        super().__init__(config)
        llm_config = AutoConfig.from_pretrained(config.llm_model_name_or_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name_or_path,
            config=llm_config,
            torch_dtype='auto',
            attn_implementation="flash_attention_2",  # or "flex_attention"
        )
        self.encoder = wenet.load_model(config.wenet_model_name_or_path)
        encoder_dim = self.encoder.encoder.output_size()
        config.hidden_size = llm_config.hidden_size  # for deepseed training
        # self.projector = ProjectorCov1d(config, encoder_dim,
        #                                 llm_config.hidden_size)
        self.projector = ProjectorConcat(encoder_dim, llm_config.hidden_size,
                                        downsample_rate=config.encoder_projector_ds_rate) # custom projector
        total_params = sum(p.numel() for p in self.projector.parameters())
        print('Projector total params: {:.2f}M'.format(total_params / 1024 /
                                                       1024))
        if config.lora_config is not None:
            lora_config = LoraConfig(**config.lora_config)
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()

        if config.pretrained_ckpt_path is not None: # load pretrained checkpoint
            self.init_weights(config.pretrained_ckpt_path)
        
        self.freeze_encoder()
        if config.lora_config is None:
            print("Freezing LLM as no LoRA config is provided.")
            self.freeze_llm()
        

    def tie_weights(self):
        return self.llm.tie_weights()

    ## for FireRedASR, we load pretrained checkpoint
    def init_weights(self, pretrained_ckpt_path: str): 
        # init weights from pretrained checkpoint, encoder has been loaded

        model_path = pretrained_ckpt_path
        projector_path = Path(model_path) / "projector.pt"
        llm_path = Path(model_path) / "llm.pt"
        self.projector.load_state_dict(torch.load(projector_path))
        self.llm.load_state_dict(torch.load(llm_path))
        
        print(f"Loaded pretrained checkpoint from {model_path} into TouchASU model.")

    def get_speech_embeddings(self, audio_features, audio_features_lengths):
        speech_emb, mask = self.encoder._forward_encoder(
            audio_features, audio_features_lengths)
        speech_emb = speech_emb.masked_fill(~mask.transpose(1, 2), 0.0)
        speech_proj = self.projector(speech_emb)
        speech_proj_lens = mask.squeeze(1).sum(1) // self.projector.k
        return speech_proj, speech_proj_lens

    def compute_mix_embedding(
        self,
        input_ids: torch.LongTensor = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
    ):
        text_emb = self.llm.get_input_embeddings()(input_ids)
        speech_emb, speech_emb_lens = self.get_speech_embeddings(
            audio_features, audio_features_lengths)
        inputs_embeds = text_emb
        for i in range(audio_features.size(0)):
            if not has_audio[i]:
                continue
            b = batch_idx[i]
            s, e = audio_offsets[i], audio_offsets[i] + speech_emb_lens[i]
            inputs_embeds[b, s:e, :] = speech_emb[i, :speech_emb_lens[i], :]
        return inputs_embeds

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_features_lengths,
            batch_idx,
            has_audio,
        )
        out = self.llm(inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask,
                       labels=labels,
                       position_ids=position_ids,
                       **kwargs)
        return out

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_features_lengths,
            batch_idx,
            has_audio,
        )
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            **kwargs,
        )
        return model_outputs

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def freeze_encoder(self):
        freeze_module(self.encoder)
        self.encoder.eval()

    def freeze_projector(self):
        freeze_module(self.projector)
        self.projector.eval()

    def freeze_llm(self):
        freeze_module(self.llm)
        self.llm.eval()

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model_name_or_path,
            padding_side="right",
        )
        # We only support QWen now
        tokenizer.bos_token = tokenizer.eos_token
        return tokenizer

class ConvDownsample(nn.Module):
    """基于Conv1d的下采样模块"""
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride=stride, padding=kernel_size//2)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # x shape: (B, T2, D2)
        x = x.transpose(1, 2)  # (B, D2, T2)
        x = self.conv(x)       # (B, output_dim, T2//stride)
        x = x.transpose(1, 2)  # (B, T2//stride, output_dim)
        x = self.norm(x)
        return x

# speaker attributed ASU model that inherits from TouchASU
class TouchASUSpeakerAttributed(TouchASU):
    model_type = 'touch_asu_speaker_attributed'
    config_class = TouchASUSpeakerAttributedConfig

    def __init__(self, config: TouchASUSpeakerAttributedConfig):
        super().__init__(config)
        self.speaker_encoder_id = config.speaker_encoder_id
        self._construct_speaker_encoder()
        self._construct_downsampler(config)

        '''
            config.projector_hidden_size: speech feature dim
            config.speaker_encoder_emb_dim: speaker embedding dim
        '''
        self.speaker_projector = nn.Sequential(
            nn.Linear(config.speaker_emb_size + config.projector_hidden_size,
                      config.projector_hidden_size),
            nn.ReLU(),
        )

        if config.asr_projector_frozen:
            logger.warning("Freezing ASR projector.")
            self.freeze_projector()
            self.projector.eval()

        if config.speaker_projector_frozen:
            logger.warning("Freezing speaker projector.")
            freeze_module(self.speaker_projector)
            self.speaker_projector.eval()

        if config.speaker_downsampler_frozen:
            logger.warning("Freezing speaker downsampler.")
            freeze_module(self.speaker_downsampler)
            self.speaker_downsampler.eval()

        self.print_trainable_parameters()

    def init_weights(self, pretrained_ckpt_path: str, pt_name: str = "model.pt"): 
        # init weights from pretrained checkpoint, encoder has been loaded
        state_dict_path = Path(pretrained_ckpt_path) / pt_name
        self.load_state_dict(torch.load(state_dict_path), strict=False)
        
        print(f"Loaded pretrained checkpoint from {state_dict_path} into TouchASU model.")

    def _construct_downsampler(self, config):
        self.speaker_downsampler = ConvDownsample(
            input_dim=config.speaker_emb_size,
            output_dim=config.speaker_emb_size,
            stride=config.speaker_downsample_rate,
            kernel_size=config.speaker_downsample_rate * 2 - 1,
        )

    def _construct_speaker_encoder(self):
        # Feature extractor
        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        
        # Load speaker encoder model
        # copied from 3D-Speaker/speakerlab/bin/infer_sv_batch.py
        conf = SV_SUPPORTS[self.speaker_encoder_id]
        cache_dir = snapshot_download(
            self.speaker_encoder_id,
            revision=conf['revision'],
        )
        pretrained_model = os.path.join(cache_dir, conf['model_pt'])
        pretrained_state = torch.load(pretrained_model, map_location='cpu')
        model_conf = conf['model']
        self.speaker_encoder = dynamic_import(model_conf['obj'])(**model_conf['args'])
        self.speaker_encoder.load_state_dict(pretrained_state)
        logger.info(f"Loaded speaker encoder: {self.speaker_encoder_id}, download directory: {pretrained_model}")

        # drop time pooling layer for utterance-level speaker embedding extraction
        if self.speaker_encoder_id == 'iic/speech_campplus_sv_zh_en_16k-common_advanced':
            # remove the last two layers: `stats` and `dense`
            # see 3D-Speaker/speakerlab/models/campplus/DTDNN.py:112
            self.speaker_encoder.xvector = self.speaker_encoder.xvector[:-2]
        else:
            raise ValueError(f"Speaker encoder {self.speaker_encoder_id} not supported yet.")
        
        freeze_module(self.speaker_encoder)
        self.speaker_encoder.eval()

    def _fuse_speaker_embedding(self, speech_emb: torch.Tensor, speaker_emb: torch.Tensor):
        ''' Speaker attributed fusion
        Args:
            speech_emb: (torch.FloatTensor) [T1, feat_dim]
            speaker_emb: (torch.FloatTensor) [T2, speaker_emb_dim]
        Returns:
            fused_emb: (torch.FloatTensor) [T1, feat_dim]
        '''
        # downsample speaker embedding to match speech_emb length
        speaker_emb = speaker_emb.unsqueeze(0)  # (1, T2, speaker)
        speaker_emb_ds = self.speaker_downsampler(speaker_emb)  # (1, T1, speaker)
        speaker_emb_ds = speaker_emb_ds.squeeze(0)  # (T1, speaker)
        
        # interpolate if lengths do not match
        if speaker_emb_ds.size(0) != speech_emb.size(0):
            speaker_emb_ds = F.interpolate(
                speaker_emb_ds.transpose(0, 1).unsqueeze(0),  # (1, speaker, T1)
                size=speech_emb.size(0),
                mode='linear',
                align_corners=False
            ).squeeze(0).transpose(0, 1)

        # concatenate and project
        fused_input = torch.cat([speech_emb, speaker_emb_ds], dim=-1)
        fused_emb = self.speaker_projector(fused_input)  # (T1, feat_dim)
        return fused_emb

    def _compute_speaker_embedding(self, raw_audio: torch.Tensor):
        ''' Compute speaker embedding from raw audio tensor
        Args:
            raw_audio: (torch.FloatTensor) [num_samples]
            Returns:
            speaker_emb: (torch.FloatTensor) [T, speaker_emb_dim]
        '''
        raw_audio = raw_audio.squeeze(0)  # (num_samples) -> (1, num_samples)
        feat = self.feature_extractor(raw_audio).unsqueeze(0)   # (1, T, 80)
        speaker_emb = self.speaker_encoder(feat).squeeze()    # (speaker_emb_dim, T)
        speaker_emb = speaker_emb.transpose(0, 1)  # (T, speaker_emb_dim)
        return speaker_emb.clone()

    def compute_mix_embedding(
        self,
        input_ids: torch.LongTensor = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        raw_audio_16k: Optional[torch.FloatTensor] = None,
        raw_audio_16k_lengths: Optional[torch.LongTensor] = None,
    ):
        text_emb = self.llm.get_input_embeddings()(input_ids)
        # speech_emb_lens \approx ceil(audio_features_lengths / 8)
        speech_emb, speech_emb_lens = self.get_speech_embeddings(
            audio_features, audio_features_lengths)
        inputs_embeds = text_emb
        for i in range(audio_features.size(0)):
            if not has_audio[i]:
                continue
            b = batch_idx[i]
            s, e = audio_offsets[i], audio_offsets[i] + speech_emb_lens[i]
            current_speech_emb = speech_emb[i, :speech_emb_lens[i], :]  # (T1, feat_dim)
            current_speaker_emb = self._compute_speaker_embedding(
                raw_audio_16k[i, :raw_audio_16k_lengths[i]]
            )  # (T2, speaker_emb_dim)
            # speaker attributed fusion
            sa_speech_emb = self._fuse_speaker_embedding(
                current_speech_emb,
                current_speaker_emb,
            )  # (T1, feat_dim)
            inputs_embeds[b, s:e, :] = sa_speech_emb
        return inputs_embeds

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,   # 这一项是west自有的变量，表示batch中是否有音频输入，与speaker attributed任务无关
        raw_audio_16k: Optional[torch.FloatTensor] = None, # raw audio for speaker encoder
        raw_audio_16k_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_features_lengths,
            batch_idx,
            has_audio,
            raw_audio_16k,
            raw_audio_16k_lengths,
        )
        out = self.llm(inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask,
                       labels=labels,
                       position_ids=position_ids,
                       **kwargs)
        return out
    
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        raw_audio_16k: Optional[torch.FloatTensor] = None, # raw audio for speaker encoder
        raw_audio_16k_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_features_lengths,
            batch_idx,
            has_audio,
            raw_audio_16k,
            raw_audio_16k_lengths,
        )
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            **kwargs,
        )
        return model_outputs

    def print_trainable_parameters(self):
        """Print trainable parameters in the model."""
        total_params = 0
        trainable_params = 0

        print("\n========== Trainable parameters (requires_grad=True) ==========")
        for name, param in self.named_parameters():
            num = param.numel()
            total_params += num
            if param.requires_grad:
                trainable_params += num
                print(f"[TRAINABLE] {name:60s} shape={tuple(param.shape)}, num_params={num}")

        print("---------------------------------------------------------------")
        print(f"Total params:     {total_params / 1e6:.2f} M")
        print(f"Trainable params: {trainable_params / 1e6:.2f} M")
        print("===============================================================\n")