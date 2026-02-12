"""
CAD-Recode Model Architecture
Paper: CAD-Recode: Reverse Engineering CAD Code from Point Clouds (ICCV 2025)
Source: https://github.com/filaPro/cad-recode

This module contains the CADRecode model class which extends Qwen2ForCausalLM
with a FourierPointEncoder for processing 3D point clouds.
"""

import torch
from torch import nn
from transformers import Qwen2ForCausalLM, Qwen2Model, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class FourierPointEncoder(nn.Module):
    """
    Encodes 3D point coordinates using Fourier features.

    Transforms (x, y, z) coordinates into high-dimensional embeddings using
    sinusoidal positional encoding, then projects to the model's hidden size.

    Input: (batch, N, 3) point coordinates
    Output: (batch, N, hidden_size) embeddings
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # 8 frequency bands: 2^0, 2^1, ..., 2^7
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies, persistent=False)
        # 3 coords + 3*8 sin + 3*8 cos = 51 dimensions
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (batch, N, 3) or (N, 3) point coordinates

        Returns:
            (batch, N, hidden_size) or (N, hidden_size) embeddings
        """
        x = points
        # Apply Fourier encoding: multiply by frequencies
        x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
        # Concatenate original coords with sin and cos features
        x = torch.cat((points, x.sin(), x.cos()), dim=-1)
        # Project to hidden size
        x = self.projection(x)
        return x


class CADRecode(Qwen2ForCausalLM):
    """
    CAD-Recode: Point cloud to CadQuery code generation model.

    Extends Qwen2-1.5B with a FourierPointEncoder to process 3D point clouds
    alongside text tokens. The model generates executable CadQuery Python code
    that reconstructs CAD models from point cloud input.

    Usage:
        model = CADRecode.from_pretrained('filapro/cad-recode-v1.5')
        # Point embeddings replace pad tokens where attention_mask == -1
    """

    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize point encoder in float32 for numerical stability
        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        point_cloud=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        """
        Forward pass with point cloud integration.

        Point cloud embeddings are inserted into the sequence where
        attention_mask == -1 (replacing pad token embeddings).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Concatenate point and text embeddings on first forward pass
        if past_key_values is None or past_key_values.get_seq_length() == 0:
            assert inputs_embeds is None
            inputs_embeds = self.model.embed_tokens(input_ids)
            point_embeds = self.point_encoder(point_cloud).bfloat16()
            # Replace pad token positions (marked with -1) with point embeddings
            inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
            attention_mask[attention_mask == -1] = 1
            input_ids = None
            position_ids = None

        # Run through transformer
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Pass point_cloud through to forward during generation."""
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs['point_cloud'] = kwargs.get('point_cloud')
        return model_inputs
