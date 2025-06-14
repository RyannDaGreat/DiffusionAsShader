from typing import Any, Dict, Optional, Tuple, Union, List, Callable

import torch, os, math
from torch import nn
from PIL import Image
from tqdm import tqdm

from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock, CogVideoXTransformer3DModel

from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline, CogVideoXPipelineOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox_video2video import CogVideoXVideoToVideoPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.models.embeddings import CogVideoXPatchEmbed
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.pipelines import DiffusionPipeline   
from diffusers.models.modeling_utils import ModelMixin

import rp

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def to_cpu(x):
    try:
        return x.to('cpu')
    except NotImplementedError:
        return x.to_empty(device='cpu').to(torch.bfloat16)
class OriginalCogVideoXTransformer3DModelTracking(CogVideoXTransformer3DModel, ModelMixin):
    """
    ORIGINAL CODE BASE - DO NOT MODIFY THIS CLASS! This exists to load the original checkpoint.

    Add tracking maps to the CogVideoX transformer model.

    Parameters:
        num_tracking_blocks (`int`, defaults to `18`):
            The number of tracking blocks to use. Must be less than or equal to num_layers.
    """

    def __init__(
        self,
        num_tracking_blocks: Optional[int] = 18,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        **kwargs
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embed_dim=time_embed_dim,
            text_embed_dim=text_embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            patch_size=patch_size,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            activation_fn=activation_fn,
            timestep_activation_fn=timestep_activation_fn,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            **kwargs
        )

        inner_dim = num_attention_heads * attention_head_dim
        self.num_tracking_blocks = num_tracking_blocks

        # Ensure num_tracking_blocks is not greater than num_layers
        if num_tracking_blocks > num_layers:
            raise ValueError("num_tracking_blocks must be less than or equal to num_layers")

        # Create linear layers for combining hidden states and tracking maps
        self.combine_linears = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim, device="cpu") for _ in range(num_tracking_blocks)]
        )

        # Initialize weights of combine_linears to zero
        for linear in self.combine_linears:
            linear.weight.data.zero_()
            linear.bias.data.zero_()

        # Create transformer blocks for processing tracking maps
        self.transformer_blocks_copy = nn.ModuleList(
            [
                to_cpu(CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    time_embed_dim=self.config.time_embed_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                ))
                for _ in range(num_tracking_blocks)
            ]
        )

        # For initial combination of hidden states and tracking maps
        self.initial_combine_linear = nn.Linear(inner_dim, inner_dim, device="cpu")
        self.initial_combine_linear.weight.data.zero_()
        self.initial_combine_linear.bias.data.zero_()

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze parameters that need to be trained
        for linear in self.combine_linears:
            for param in linear.parameters():
                param.requires_grad = True
        
        for block in self.transformer_blocks_copy:
            for param in block.parameters():
                param.requires_grad = True
        
        for param in self.initial_combine_linear.parameters():
            param.requires_grad = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        tracking_maps: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        # Process tracking maps
        prompt_embed = encoder_hidden_states.clone()
        tracking_maps_hidden_states = self.patch_embed(prompt_embed, tracking_maps)
        tracking_maps_hidden_states = self.embedding_dropout(tracking_maps_hidden_states)
        del prompt_embed

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        tracking_maps = tracking_maps_hidden_states[:, text_seq_length:]

        # Combine hidden states and tracking maps initially
        combined = hidden_states + tracking_maps
        tracking_maps = self.initial_combine_linear(combined)

        # Process transformer blocks
        for i in range(len(self.transformer_blocks)):
            if self.training and self.gradient_checkpointing:
                # Gradient checkpointing logic for hidden states
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.transformer_blocks[i]),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = self.transformer_blocks[i](
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
            
            if i < len(self.transformer_blocks_copy):
                if self.training and self.gradient_checkpointing:
                    # Gradient checkpointing logic for tracking maps
                    tracking_maps, _ = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.transformer_blocks_copy[i]),
                        tracking_maps,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    tracking_maps, _ = self.transformer_blocks_copy[i](
                        hidden_states=tracking_maps,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )
                
                # Combine hidden states and tracking maps
                tracking_maps = self.combine_linears[i](tracking_maps)
                hidden_states = hidden_states + tracking_maps
                

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # try:
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            print("Loaded DiffusionAsShader checkpoint directly.")
            
            for param in model.parameters():
                param.requires_grad = False
            for linear in model.combine_linears:
                for param in linear.parameters():
                    param.requires_grad = True
                
            for block in model.transformer_blocks_copy:
                for param in block.parameters():
                    param.requires_grad = True
                
            for param in model.initial_combine_linear.parameters():
                param.requires_grad = True
            
            return model
        
        # except Exception as e:
        #     print(f"Failed to load as DiffusionAsShader: {e}")
        #     print("Attempting to load as CogVideoXTransformer3DModel and convert...")

        #     base_model = CogVideoXTransformer3DModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
            
        #     config = dict(base_model.config)
        #     config["num_tracking_blocks"] = kwargs.pop("num_tracking_blocks", 18)
            
        #     model = cls(**config)
        #     model.load_state_dict(base_model.state_dict(), strict=False)

        #     model.initial_combine_linear.weight.data.zero_()
        #     model.initial_combine_linear.bias.data.zero_()
            
        #     for linear in model.combine_linears:
        #         linear.weight.data.zero_()
        #         linear.bias.data.zero_()
            
        #     for i in range(model.num_tracking_blocks):
        #         model.transformer_blocks_copy[i].load_state_dict(model.transformer_blocks[i].state_dict())
            

        #     for param in model.parameters():
        #         param.requires_grad = False
            
        #     for linear in model.combine_linears:
        #         for param in linear.parameters():
        #             param.requires_grad = True
                
        #     for block in model.transformer_blocks_copy:
        #         for param in block.parameters():
        #             param.requires_grad = True
                
        #     for param in model.initial_combine_linear.parameters():
        #         param.requires_grad = True
            
        #     return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "5GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            save_function=save_function,
            safe_serialization=safe_serialization,
            variant=variant,
            max_shard_size=max_shard_size,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        
        if is_main_process:
            config_dict = dict(self.config)
            config_dict.pop("_name_or_path", None)
            config_dict.pop("_use_default_values", None)
            config_dict["_class_name"] = "OriginalCogVideoXTransformer3DModelTracking"
            config_dict["num_tracking_blocks"] = self.num_tracking_blocks
            
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(config_dict, f, indent=2)

class CogVideoXTransformer3DModelTracking(CogVideoXTransformer3DModel, ModelMixin):
    """
    Add tracking maps to the CogVideoX transformer model.

    Parameters:
        num_tracking_blocks (`int`, defaults to `18`):
            The number of tracking blocks to use. Must be less than or equal to num_layers.
    """

    def __init__(
        self,
        num_tracking_blocks: Optional[int] = 18,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        **kwargs
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embed_dim=time_embed_dim,
            text_embed_dim=text_embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            patch_size=patch_size,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            activation_fn=activation_fn,
            timestep_activation_fn=timestep_activation_fn,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            **kwargs
        )

        inner_dim = num_attention_heads * attention_head_dim
        self.num_tracking_blocks = num_tracking_blocks

        # Ensure num_tracking_blocks is not greater than num_layers
        if num_tracking_blocks > num_layers:
            raise ValueError("num_tracking_blocks must be less than or equal to num_layers")

        # Create linear layers for combining hidden states and tracking maps
        self.combine_linears = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim) for _ in range(num_tracking_blocks)]
        )

        # Initialize weights of combine_linears to zero
        for linear in self.combine_linears:
            linear.weight.data.zero_()
            linear.bias.data.zero_()

        # Create transformer blocks for processing tracking maps
        self.transformer_blocks_copy = nn.ModuleList(
            [
                to_cpu(CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    time_embed_dim=self.config.time_embed_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                ))
                for _ in range(num_tracking_blocks)
            ]
        )

        # For initial combination of hidden states and tracking maps
        self.initial_combine_linear = nn.Linear(inner_dim, inner_dim)
        self.initial_combine_linear.weight.data.zero_()
        self.initial_combine_linear.bias.data.zero_()

        # HERE'S THE PICKLE: How can I load checkpoints that don't have this?? Strict=false or somethin?  OR do we just add this on after teh fact and save it and load it or something?
        # Z. Patch embedding 2, for the control branch...
        # Copied from original cogvid code
        self.second_patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=None, #GUESSWORK
            in_channels=in_channels * 3, #THREE: For all 3 control videos: tracking, counter_tracking and counter_video
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=True, #GUESSWORK
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze parameters that need to be trained
        for linear in self.combine_linears:
            for param in linear.parameters():
                param.requires_grad = True
        
        for block in self.transformer_blocks_copy:
            for param in block.parameters():
                param.requires_grad = True
        
        for param in self.initial_combine_linear.parameters():
            param.requires_grad = True
        
        for param in self.second_patch_embed.proj.parameters():
            param.requires_grad = True

    def forward(
        self,
        hidden_states: torch.Tensor,         # BTCHW
        encoder_hidden_states: torch.Tensor, # B Seq_Len Dim

        tracking_maps: torch.Tensor,         # BTCHW
        counter_tracking_maps: torch.Tensor, # BTCHW
        counter_video_maps: torch.Tensor,    # BTCHW

        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        #FOR REFERENCE:
        # text_embeds = encoder_hidden_states (BSD) batch seq_len dim
        # image_embeds = hidden_states (BCTHW) batch chan time height width
        #Channelwise-concatenation of all 3 control signals

        control_maps = torch.cat(
            [
                tracking_maps,
                counter_tracking_maps,
                counter_video_maps,
            ],
            dim=2,
        )


        rp.validate_tensor_shapes(
            hidden_states         = "B T C H W",
            #
            tracking_maps         = "B T C H W",
            counter_tracking_maps = "B T C H W",
            counter_video_maps    = "B T C H W",
            #
            control_maps          = "B T CCC H W",
            #
            encoder_hidden_states = "B Seq Dim",
            #
            C=32,
            CCC=32*3,
            #
            verbose=False,
            # verbose='bold altbw white random blue',
        )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        # Transform latents [B, F, C, H, W] into tokens and concatenate with text embeddings
        # Result shape: [batch_size, text_seq_length + num_patches, embed_dim]
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states) #forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor)
        hidden_states = self.embedding_dropout(hidden_states)

        #BEFORE PATCH EMBED, UGH THIS CODE DRIVES ME MAD....WHY DID THEY CALL IT HIDDEN_STATES....

        # Process tracking maps - apply the same transformation
        prompt_embed = encoder_hidden_states.clone()
        tracking_maps_hidden_states = self.second_patch_embed(prompt_embed, control_maps)
        tracking_maps_hidden_states = self.embedding_dropout(tracking_maps_hidden_states)
        del prompt_embed

        # At this point, hidden_states contains a concatenation of text tokens and image tokens
        # We need to separate them to process text and image differently
        text_seq_length = encoder_hidden_states.shape[1]
        
        # Get the text tokens from the beginning of the sequence
        encoder_hidden_states = hidden_states[:, :text_seq_length]  # Shape: [batch_size, text_seq_length, embed_dim]
        
        # Get the image tokens (everything after the text tokens)
        hidden_states = hidden_states[:, text_seq_length:]  # Shape: [batch_size, num_patches, embed_dim]
        
        # Apply the same separation to tracking maps
        tracking_maps = tracking_maps_hidden_states[:, text_seq_length:]  # Keep only image tokens

        # Combine hidden states and tracking maps initially
        combined = hidden_states + tracking_maps
        tracking_maps = self.initial_combine_linear(combined)

        # Process transformer blocks
        for i in range(len(self.transformer_blocks)):
            if self.training and self.gradient_checkpointing:
                # Gradient checkpointing logic for hidden states
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.transformer_blocks[i]),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = self.transformer_blocks[i](
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
            
            if i < len(self.transformer_blocks_copy):
                if self.training and self.gradient_checkpointing:
                    # Gradient checkpointing logic for tracking maps
                    tracking_maps, _ = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.transformer_blocks_copy[i]),
                        tracking_maps,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    tracking_maps, _ = self.transformer_blocks_copy[i](
                        hidden_states=tracking_maps,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )
                
                # Combine hidden states and tracking maps
                tracking_maps = self.combine_linears[i](tracking_maps)
                hidden_states = hidden_states + tracking_maps
                

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        if 'T2V_TRANSFORMER_CHECKPOINT' in os.environ:
            t2v_path = os.environ['T2V_TRANSFORMER_CHECKPOINT']
            rp.fansi_print(f'USING T2V CHECKPOINT TOO! t2v_path={t2v_path}','white blue cyan bold underlined on dark dark gray yellow green')
        try:
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            print("Loaded DiffusionAsShader checkpoint directly.")

            def i2v_to_t2v(transformer, t2v_transformer_root="/home/jupyter/CleanCode/Huggingface/CogVideoX-5b/transformer"):
                # t2v_transformer_root = "/home/jupyter/CleanCode/Checkpoints/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans_RandomSpeed_WithDropout_2500_10000000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-14700/transformer"
                # t2v_transformer_root = "/home/jupyter/CleanCode/Huggingface/CogVideoX-5b/transformer"
                safetensor_paths = rp.get_all_files(t2v_transformer_root, file_extension_filter="safetensors")
                
                # Announce
                rp.fansi_print(
                    f"i2v_to_t2v: Loading Safetensors from {t2v_transformer_root}\n"
                    + rp.indentify(
                        rp.line_join(rp.get_relative_paths(safetensor_paths, root=t2v_transformer_root)), "    • "
                    ),
                    "light gray cyan blue",
                    truecolor=True,
                )
                
                merged_safetensors= rp.merged_dicts(
                    [rp.load_safetensors(x) for x in rp.eta(safetensor_paths, "Loading Safetensors")]
            )
                
                #Correct the number of channels in the T2V model if needed; by padding zeros - effecively turning it into a T2V model
                assert 'patch_embed.proj.weight' in merged_safetensors
                weight = merged_safetensors['patch_embed.proj.weight']
                if list(weight.shape)==[3072, 16, 2, 2]:
                    new_weight = torch.zeros([3072, 32, 2, 2], dtype=weight.dtype, device=weight.device)
                    new_weight[:,:16] = weight
                    merged_safetensors['patch_embed.proj.weight'] = new_weight
                    rp.fansi_print(f"i2v_to_t2v: Expanded patch_embed.proj.weight from {weight.shape} to {new_weight.shape}", 'yellow')
                
                if hasattr(transformer,'_orig_mod'):
                    #https://github.com/karpathy/nanoGPT/issues/325
                    rp.fansi_print('i2v_to_t2v: Using transformer._orig_mod','yellow')
                    transformer=transformer._orig_mod
                
                #OK...load in the T2V weights!
                status = transformer.load_state_dict(merged_safetensors, strict=False)
                
                #IMPORTANT! If we're switching to CogX T2V from an I2V model you have to do this!
                transformer.patch_embed.use_learned_positional_embeddings=False
                rp.fansi_print(f"i2v_to_t2v: Disabled use_learned_positional_embeddings!", 'yellow')
                
                rp.fansi_print(f'i2v_to_t2v: {status}','yellow')


# TODO: Something like this if we want to be able to initialize from original T2V model
#           def revert_state_dict():
#                 for key in state:
#                     if '_copy' in key:
#                         #EXAMPLE: transformer_blocks_copy.17.norm1.norm.weight
#                         state[key]=state[key.replace('_copy','')]
#                         print(f'Replacing {key}')
#                     if 'combine' in key:
#                         state[key].zero_()
#                         print(f'Zeroing {key}')
#
#             state=load_safetensors('*.safetensors',keys_only=False)
#             revert_state_dict(state)"""
#
#             def maybe_zero(param):
#                 if os.environ.get('DISABLE_CONTROLNET'):
#                     param.zero_()
#                     rp.fansi_print(
#                         f"DISABLE_CONTROLNET: Zeroing - {(param**2).sum()}",
#                         "red red bold on yellow yellow",
#                     )

            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze parameters that need to be trained
            for block in model.transformer_blocks_copy:
                for param in block.parameters():
                    param.requires_grad = True
                
            for linear in model.combine_linears:
                for param in linear.parameters():
                    # maybe_zero(param)
                    param.requires_grad = True
                
            for param in model.initial_combine_linear.parameters():
                # maybe_zero(param)
                param.requires_grad = True
        
            for param in model.second_patch_embed.proj.parameters():
                param.requires_grad = True
            
            if 't2v_path' in vars():
                rp.fansi_print(f'LOADING T2V CHECKPOINT OVERRIDE: t2v_path={t2v_path}','white blue cyan bold underlined on dark dark gray yellow green')
                i2v_to_t2v(model)

                #Super important we do this!
                model.patch_embed       .use_learned_positional_embeddings=False
                model.second_patch_embed.use_learned_positional_embeddings=False

            return model 
        
        except Exception as e:
            # assert False, f'We should be initializing from a DiffusionAsShader checkpoint, not an initial I2V checkpoint... pretrained_model_name_or_path = {pretrained_model_name_or_path}'

            # pretrained_model_name_or_path = '/home/jupyter/CleanCode/Huggingface/CogVideoX-5b-I2V'
            rp.fansi_print(f"Failed to load as DiffusionAsShader: {e}",'yellow white on black black green bold')
            rp.fansi_print(
                f"Attempting to load as CogVideoXTransformer3DModel and convert from pretrained_model_name_or_path={repr(pretrained_model_name_or_path)} (i.e. {repr(rp.get_absolute_path(pretrained_model_name_or_path))})...",
                "yellow white on black black green bold",
            )

            base_model = OriginalCogVideoXTransformer3DModelTracking.from_pretrained(pretrained_model_name_or_path, **kwargs)
            
            config = dict(base_model.config)
            
            model = cls(**config)
            model.load_state_dict(base_model.state_dict(), strict=False)

            model.second_patch_embed.proj.weight.data.zero_()
            model.second_patch_embed.proj.bias.data.zero_()
            model.second_patch_embed.text_proj.load_state_dict(model.patch_embed.text_proj.state_dict())
            
            #How I came up with the next lines of code:
            #        >>> if not 'm' in vars():
            #        ...     import models.cogvideox_tracking as t
            #        ...     m=t.CogVideoXTransformer3DModelTracking.from_pretrained('/home/jupyter/CleanCode/Github/DiffusionAsShader/diffusion_shader_model/transformer')
            #        ...     print(m.transformer_blocks_copy[5].attn1.to_v.weight.data[0])
            #        ... 
            #        ... from icecream import ic
            #        ... ic(m.patch_embed.text_proj.weight.data)
            #        ... ic(m.second_patch_embed.text_proj.weight.data)
            #        ... # ┌                                                                                         ┐
            #        ... # │┌                                                                                 ┐    ┌┐│
            #        ... ic((m.second_patch_embed.text_proj.weight.data == m.patch_embed.text_proj.weight.data).all())
            #        ... # │└                                                                                 ┘    └┘│
            #        ... # └                                                                                         ┘
            #        ... ic(m.second_patch_embed.proj.weight.data.any())
            #        ... ic(
            #        ...     m.patch_embed.proj.bias.shape,
            #        ...     m.second_patch_embed.proj.bias.shape,
            #        ...     m.patch_embed.proj.bias.shape == m.second_patch_embed.proj.bias.shape,
            #        ... )
            #        ... ic(
            #        ...     m.patch_embed.proj.weight.data.shape,
            #        ...     m.second_patch_embed.proj.weight.data.shape,
            #        ... )
            #        ic| m.patch_embed.text_proj.weight.data: tensor([[-0.0042,  0.0457,  0.0121,  ...,  0.0091,  0.0104, -0.0090],
            #                                                         [-0.0096,  0.0042,  0.0270,  ..., -0.0388, -0.0500,  0.0229],
            #                                                         [ 0.0058, -0.0046, -0.0598,  ..., -0.0225,  0.0164,  0.0093],
            #                                                         ...,
            #                                                         [ 0.0304,  0.0103,  0.0097,  ..., -0.0156, -0.0242,  0.0608],
            #                                                         [ 0.0298, -0.0126,  0.0210,  ...,  0.0232,  0.0161, -0.0165],
            #                                                         [-0.0001, -0.0119, -0.0025,  ...,  0.0113, -0.0549,  0.0198]])
            #        ic| m.second_patch_embed.text_proj.weight.data: tensor([[-0.0042,  0.0457,  0.0121,  ...,  0.0091,  0.0104, -0.0090],
            #                                                                [-0.0096,  0.0042,  0.0270,  ..., -0.0388, -0.0500,  0.0229],
            #                                                                [ 0.0058, -0.0046, -0.0598,  ..., -0.0225,  0.0164,  0.0093],
            #                                                                ...,
            #                                                                [ 0.0304,  0.0103,  0.0097,  ..., -0.0156, -0.0242,  0.0608],
            #                                                                [ 0.0298, -0.0126,  0.0210,  ...,  0.0232,  0.0161, -0.0165],
            #                                                                [-0.0001, -0.0119, -0.0025,  ...,  0.0113, -0.0549,  0.0198]])
            #        ic| (m.second_patch_embed.text_proj.weight.data == m.patch_embed.text_proj.weight.data).all(): tensor(True)
            #        ic| m.second_patch_embed.proj.weight.data.any(): tensor(False)
            #        ic| m.patch_embed.proj.bias.shape: torch.Size([3072])
            #            m.second_patch_embed.proj.bias.shape: torch.Size([3072])
            #            m.patch_embed.proj.bias.shape == m.second_patch_embed.proj.bias.shape: True
            #        ic| m.patch_embed.proj.weight.data.shape: torch.Size([3072, 32, 2, 2])
            #            m.second_patch_embed.proj.weight.data.shape: torch.Size([3072, 96, 2, 2])
            model.second_patch_embed.proj.bias  .data[:]        =model.patch_embed.proj.bias  .data
            model.second_patch_embed.proj.weight.data[:,:32,:,:]=model.patch_embed.proj.weight.data

            for param in model.parameters():
                param.requires_grad = False
            
            for linear in model.combine_linears:
                for param in linear.parameters():
                    param.requires_grad = True
                
            for block in model.transformer_blocks_copy:
                for param in block.parameters():
                    param.requires_grad = True
                
            for param in model.initial_combine_linear.parameters():
                param.requires_grad = True
        
            for param in model.second_patch_embed.proj.parameters():
                param.requires_grad = True
            
            return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "5GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            save_function=save_function,
            safe_serialization=safe_serialization,
            variant=variant,
            max_shard_size=max_shard_size,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        
        if is_main_process:
            config_dict = dict(self.config)
            config_dict.pop("_name_or_path", None)
            config_dict.pop("_use_default_values", None)
            config_dict["_class_name"] = "CogVideoXTransformer3DModelTracking"
            config_dict["num_tracking_blocks"] = self.num_tracking_blocks
            
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(config_dict, f, indent=2)

class CogVideoXImageToVideoPipelineTracking(CogVideoXImageToVideoPipeline, DiffusionPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModelTracking,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        
        if not isinstance(self.transformer, CogVideoXTransformer3DModelTracking):
            raise ValueError("The transformer in this pipeline must be of type CogVideoXTransformer3DModelTracking")
            
        # 打印transformer blocks的数量
        print(f"Number of transformer blocks: {len(self.transformer.transformer_blocks)}")
        print(f"Number of tracking transformer blocks: {len(self.transformer.transformer_blocks_copy)}")
        self.transformer = torch.compile(self.transformer)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        tracking_maps: torch.Tensor=None,
        tracking_image: torch.Tensor=None,
        counter_tracking_maps: torch.Tensor=None,
        counter_tracking_image: torch.Tensor=None,
        counter_video_maps: torch.Tensor=None,
        counter_video_image: torch.Tensor=None,
        use_image_conditioning = True,
        latent_conditioning_dropout = [1,1,1,1,1,1,1,1,1,1,1,1,1],
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        # Most of the implementation remains the same as the parent class
        # We will modify the parts that need to handle tracking_maps

        assert tracking_maps is not None
        assert tracking_image is not None
        assert counter_tracking_maps is not None
        assert counter_tracking_image is not None
        assert counter_video_maps is not None
        assert counter_video_image is not None
        assert len(latent_conditioning_dropout) == 13


        # 1. Check inputs and set default values
        self.check_inputs(
            image,
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            del negative_prompt_embeds

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        # Process the input image - the image to use as first frame reference [B, C, H, W]
        # This is a single image (e.g., the first frame of the desired video)

        image                  = self.video_processor.preprocess(image                 , height=height, width=width).to(device, dtype=prompt_embeds.dtype)
        tracking_image         = self.video_processor.preprocess(tracking_image        , height=height, width=width).to(device, dtype=prompt_embeds.dtype)
        counter_tracking_image = self.video_processor.preprocess(counter_tracking_image, height=height, width=width).to(device, dtype=prompt_embeds.dtype)
        counter_video_image    = self.video_processor.preprocess(counter_video_image   , height=height, width=width).to(device, dtype=prompt_embeds.dtype)

        if not use_image_conditioning:
            image = image * 0
            counter_video_image = counter_video_image * 0

        if self.transformer.config.in_channels != 16:
            latent_channels = self.transformer.config.in_channels // 2
        else:
            latent_channels = self.transformer.config.in_channels
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        _, tracking_image_latents = self.prepare_latents(
            tracking_image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )

        # 2. Process counter tracking maps
        # Get latents for counter tracking first frame
        _, counter_tracking_image_latents = self.prepare_latents(
            counter_tracking_image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )
                        
        # 3. Process counter video maps
        # Get latents for counter video first frame
        _, counter_video_image_latents = self.prepare_latents(
            counter_video_image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )

        for frame_number, keep in enumerate(latent_conditioning_dropout):
            if not keep:
                print("Discarding counterfactual frame",frame_number, 'with shape',counter_video_maps.shape)
                counter_video_maps[:,frame_number] = 0

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loopx
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Main content input - the noisy latents being denoised [B, T, C, H, W]
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Image conditioning - the first frame latents [B, 1, C, H, W]
                # Used to condition the generation to match the first frame
                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                
                # Concatenate along channel dimension (dim=2)
                # Result: [B, T, 2*C, H, W] where the channels contain both content and conditioning
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

                # Handle all three tracking map streams (all required inputs)
                
                # 1. Process primary tracking maps
                latents_tracking_image = torch.cat([tracking_image_latents] * 2) if do_classifier_free_guidance else tracking_image_latents
                tracking_maps_input = torch.cat([tracking_maps] * 2) if do_classifier_free_guidance else tracking_maps
                tracking_maps_input = torch.cat([tracking_maps_input, latents_tracking_image], dim=2)
                
                latents_counter_tracking_image = torch.cat([counter_tracking_image_latents] * 2) if do_classifier_free_guidance else counter_tracking_image_latents
                counter_tracking_maps_input = torch.cat([counter_tracking_maps] * 2) if do_classifier_free_guidance else counter_tracking_maps
                counter_tracking_maps_input = torch.cat([counter_tracking_maps_input, latents_counter_tracking_image], dim=2)

                
                latents_counter_video_image = torch.cat([counter_video_image_latents] * 2) if do_classifier_free_guidance else counter_video_image_latents
                counter_video_maps_input = torch.cat([counter_video_maps] * 2) if do_classifier_free_guidance else counter_video_maps
                counter_video_maps_input = torch.cat([counter_video_maps_input, latents_counter_video_image], dim=2)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Predict noise
                self.transformer.to(dtype=latent_model_input.dtype)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    tracking_maps=tracking_maps_input,
                    counter_tracking_maps=counter_tracking_maps_input,
                    counter_video_maps=counter_video_maps_input,
                    return_dict=False,
                )[0]
                
                # Clean up tensors to free memory
                # del latent_model_input
                # del tracking_maps_input
                # del counter_tracking_maps_input
                # del counter_video_maps_input
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    del noise_pred_uncond, noise_pred_text

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                del noise_pred
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 9. Post-processing
        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
