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

# for diar attributed ASU
import traceback
try:
    from nemo.collections.asr.models import SortformerEncLabelModel
except ImportError:
    traceback.print_exc()
    raise ImportError("Please install NeMo to use SortformerEncLabelModel.")


from .configuration_touch_asu import TouchASUWithDiarModuleConfig
import torch.nn.functional as F

from transformers.utils import logging
logger = logging.get_logger(__name__)

class ProjectorCov1d(nn.Module):

    def __init__(self, ds_rate, projector_hidden_dim, encoder_dim, llm_dim):
        super().__init__()
        self.k = ds_rate
        self.conv1d = nn.Conv1d(in_channels=encoder_dim,
                                out_channels=encoder_dim,
                                kernel_size=self.k,
                                stride=self.k,
                                padding=0)
        self.linear1 = nn.Linear(encoder_dim, projector_hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(projector_hidden_dim, llm_dim)
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
        #   llm_config.hidden_size)
        # self.projector = ProjectorCov1d(
        #   ds_rate=config.encoder_projector_ds_rate,
        #   projector_hidden_dim=config.projector_hidden_size,
        #   encoder_dim=encoder_dim,
        #   llm_dim=llm_config.hidden_size)
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

# ASU model with an SD module that inherits from TouchASU
class TouchASUWithDiarModule(TouchASU):
    model_type = 'touch_asu_with_diar_module'
    config_class = TouchASUWithDiarModuleConfig

    def __init__(self, config: TouchASUWithDiarModuleConfig):
        super().__init__(config)
    
        self.diar_projector = ProjectorCov1d(
            ds_rate=config.diar_downsample_rate,
            encoder_dim=config.diar_emb_size,
            projector_hidden_dim=config.projector_hidden_size,
            llm_dim=config.projector_hidden_size,
        )
        
        self.diar_model = SortformerEncLabelModel.restore_from(
            restore_path=config.diar_model_name_or_path,
            map_location='cuda',
            strict=False,
        )

        if config.asr_projector_frozen:
            logger.warning("Freezing ASR projector.")
            self.freeze_projector()
            self.projector.eval()

        if config.diar_projector_frozen:
            logger.warning("Freezing diar downsampler.")
            freeze_module(self.diar_projector)
            self.diar_projector.eval()

        if config.diar_module_frozen:
            logger.warning("Freezing diar downsampler.")
            freeze_module(self.diar_model)
            self.diar_model.eval()

        self.print_trainable_parameters()

    def init_weights(self, pretrained_ckpt_path: str, pt_name: str = "model.pt"): 
        # init weights from pretrained checkpoint, encoder has been loaded
        state_dict_path = Path(pretrained_ckpt_path) / pt_name
        self.load_state_dict(torch.load(state_dict_path), strict=False)
        
        print(f"Loaded pretrained checkpoint from {state_dict_path} into TouchASU model.")



    def compute_mix_embedding(
        self,
        input_ids: torch.LongTensor = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        raw_audio_path: Optional[list[str]] = None, # raw audio for speaker encoder
        eosd_to_soa: Optional[torch.LongTensor] = None,
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
            
            # audio
            s, e = audio_offsets[i], audio_offsets[i] + speech_emb_lens[i]
            # sd
            # TODO: make diarize support gradient calculation
            _, predicted_probs = self.diar_model.diarize(
                audio=raw_audio_path[i],
                batch_size=1,
                include_tensor_outputs=True
            )
            sd_probs = predicted_probs[0] # list[torch.Tensor] -> torch.Tensor (1, T, 4)
            sd_probs = sd_probs.clone().to(audio_features)
            sd_emb = self.diar_projector(sd_probs)  # (1, T', feat_dim)
            e_sd = s - eosd_to_soa[i] + 1
            sd_feature_length = sd_emb.shape[1] # T'
            s_sd = e_sd - sd_feature_length
            # print(s_sd, e_sd, sd_feature_length)

            # replace placeholder with sd and audio embeddings into inputs_embeds
            inputs_embeds[b, s_sd:e_sd, :] = sd_emb[0]
            inputs_embeds[b, s:e, :] = speech_emb[i, :speech_emb_lens[i], :]  # (T1, feat_dim)
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
        raw_audio_path: Optional[list[str]] = None, # raw audio for speaker encoder
        eosd_to_soa: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_features_lengths,
            batch_idx,
            has_audio,
            raw_audio_path,
            eosd_to_soa,
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
        has_audio: Optional[torch.BoolTensor] = None,   # 这一项是west自有的变量，表示batch中是否有音频输入，与speaker attributed任务无关
        raw_audio_path: Optional[list[str]] = None, # raw audio for speaker encoder
        eosd_to_soa: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_features_lengths,
            batch_idx,
            has_audio,
            raw_audio_path,
            eosd_to_soa,
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