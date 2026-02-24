"""
Cadrille Model Definition

Multi-modal CAD reconstruction model based on Qwen2-VL.
Supports point cloud, image, and text inputs.

Based on: https://github.com/col14m/cadrille
Paper: "Cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning"
"""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, List, Union

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    print("[Cadrille] Warning: qwen_vl_utils not installed. Image mode will not work.")

try:
    from transformers import Qwen2VLForConditionalGeneration
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[Cadrille] Warning: transformers not installed.")


class FourierEmbedder(nn.Module):
    """
    Fourier feature embedding for 3D coordinates.
    Maps input coordinates to higher-dimensional space using sinusoidal functions.
    """
    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        include_input: bool = True,
        include_pi: bool = True
    ):
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer('frequencies', frequencies, persistent=False)
        self.include_input = include_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., 3) for 3D coordinates

        Returns:
            Fourier embedded tensor
        """
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class FourierPointEncoder(nn.Module):
    """
    Encodes 3D point clouds using Fourier feature embedding.
    Projects to model hidden size for integration with language model.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # 8 frequencies, no pi multiplication (as in original Cadrille)
        self.fourier_embedder = FourierEmbedder(num_freqs=8, include_pi=False)
        # Input dim: 3 (xyz) + 8*2*3 (sin/cos for each freq and coord) = 51
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: Point cloud tensor of shape (batch, n_points, 3)

        Returns:
            Encoded points of shape (batch, n_points, hidden_size)
        """
        x = self.fourier_embedder(points[..., :3])
        x = self.projection(x)
        return x


if HAS_TRANSFORMERS:
    class Cadrille(Qwen2VLForConditionalGeneration):
        """
        Cadrille: Multi-modal CAD reconstruction model.

        Extends Qwen2-VL with a point cloud encoder for multi-modal CAD generation.
        Supports:
        - Point cloud input (256 points)
        - Image input (4 orthogonal views)
        - Text input (descriptions)

        Output: CadQuery Python code
        """

        def __init__(self, config):
            super().__init__(config)

            # Initialize point encoder in float32 for stability
            torch.set_default_dtype(torch.float32)
            self.point_encoder = FourierPointEncoder(config.hidden_size)
            torch.set_default_dtype(torch.bfloat16)

            # Cache for RoPE deltas (needed for position encoding)
            self.rope_deltas = None

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.Tensor] = None,
            video_grid_thw: Optional[torch.Tensor] = None,
            rope_deltas: Optional[torch.Tensor] = None,
            cache_position: Optional[torch.Tensor] = None,
            point_clouds: Optional[torch.Tensor] = None,
            is_pc: Optional[torch.Tensor] = None,
            is_img: Optional[torch.Tensor] = None,
        ):
            """
            Forward pass with multi-modal support.

            Additional args beyond Qwen2VL:
                point_clouds: (batch, n_points, 3) point cloud coordinates
                is_pc: (batch,) boolean mask for point cloud inputs
                is_img: (batch,) boolean mask for image inputs
            """
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if inputs_embeds is None:
                inputs_embeds = self.model.language_model.embed_tokens(input_ids)

                # Process image inputs
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.visual.get_dtype())
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                    n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )
                    image_mask = (
                        (input_ids == self.config.image_token_id)
                        .unsqueeze(-1)
                        .expand_as(inputs_embeds)
                        .to(inputs_embeds.device)
                    )
                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                # Process video inputs (used for multi-view images)
                if is_img is not None and is_img.sum() > 0 and pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos[is_img]
                    pixel_values_videos = pixel_values_videos.view(-1, pixel_values_videos.shape[-1])
                    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                    video_grid_thw = video_grid_thw[is_img]
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                    n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                    n_video_features = video_embeds.shape[0]
                    if n_video_tokens != n_video_features:
                        raise ValueError(
                            f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                        )
                    video_mask = (
                        (input_ids == self.config.video_token_id)
                        .unsqueeze(-1)
                        .expand_as(inputs_embeds)
                        .to(inputs_embeds.device)
                    )
                    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                # Process point cloud inputs
                if is_pc is not None and is_pc.sum() > 0 and (past_key_values is None or past_key_values.get_seq_length() == 0):
                    point_embeds = self.point_encoder(point_clouds.float()).to(inputs_embeds.dtype)
                    start_idxs = attention_mask.shape[1] - attention_mask.sum(axis=1)
                    for i, start_idx in enumerate(start_idxs):
                        if is_pc[i]:
                            inputs_embeds[i, start_idx:start_idx + point_embeds.shape[1], :] = point_embeds[i]

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

            # Calculate position IDs with RoPE
            if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                ):
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids, image_grid_thw, video_grid_thw, attention_mask
                    )
                    self.rope_deltas = rope_deltas
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                        delta = delta.to(position_ids.device)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
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

            loss = None
            if labels is not None:
                logits = logits.float()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return Qwen2VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
            )

        def prepare_inputs_for_generation(self, *args, **kwargs):
            """Prepare inputs for generation, including multi-modal inputs."""
            model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
            model_inputs['point_clouds'] = kwargs.get('point_clouds')
            model_inputs['is_pc'] = kwargs.get('is_pc')
            model_inputs['is_img'] = kwargs.get('is_img')
            return model_inputs

else:
    # Stub class if transformers not available
    class Cadrille:
        def __init__(self, *args, **kwargs):
            raise ImportError("transformers package is required for Cadrille model")
