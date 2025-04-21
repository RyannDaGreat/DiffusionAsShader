# DiffusionAsShader Repository Guide

This document provides a high-level overview of the DiffusionAsShader repository structure, key components, and important functionality to help with navigation and understanding.

## Repository Overview

DiffusionAsShader is a 3D-aware video diffusion model for versatile video generation control, allowing for:
- Motion Transfer
- Camera Control (panning, zooming, rotation)
- Object Manipulation
- Animating 3D meshes to realistic videos

## Model Architecture

DiffusionAsShader modifies the CogVideoX architecture by adding a parallel "tracking stream" that processes tracking maps and injects activations into the main content stream.

### Input Tensors and Their Dimensions

The model uses several key tensors as input:

1. **image**:
   - A single input image (e.g., the first frame of the desired video)
   - Dimension: [B, C, H, W] (batch, channels, height, width)
   - Purpose: Serves as the reference for the first frame of the generated video
   - After encoding → **image_latents**

2. **tracking_image**:
   - The first frame of the tracking video
   - Dimension: [B, C, H, W]
   - Purpose: Serves as a reference/conditioning for the tracking motion
   - After encoding → **tracking_image_latents**

3. **tracking_maps**:
   - The full tracking video showing motion paths (colored line visualization)
   - Initial dimension: [B, T, C, H, W] (batch, time/frames, channels, height, width)
   - After encoding: [B, F, C, H, W] (where C is the latent channels)
   - Purpose: Provides the motion guidance for the generation process

During the denoising loop, both streams follow the same pattern:
- Main stream: `latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)`
- Tracking stream: `tracking_maps_input = torch.cat([tracking_maps_input, latents_tracking_image], dim=2)`

This concatenation doubles the channel dimension for both streams, creating a parallel structure where:
- The main stream processes noisy content latents conditioned on the first frame
- The tracking stream processes tracking motion conditioned on the first tracking frame

### Key Model Parameters
- Base model: CogVideoX-5b-I2V
- Input channels: 16 channels for the transformer model (`in_channels` parameter)
- For Image-to-Video (I2V), the image conditioning is handled by:
  ```python
  # First, prepare latents for video and image conditioning
  latents, image_latents = self.prepare_latents(...)
  
  # During each denoising step:
  latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
  latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
  
  # Concatenate along dim=2 (channel dimension)
  latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
  ```
  
  This effectively doubles the channel dimension by concatenating the image latents.

- The tracking conditioning works the same way:
  ```python
  # For tracking conditioning
  latents_tracking_image = torch.cat([tracking_image_latents] * 2) if do_classifier_free_guidance else tracking_image_latents
  tracking_maps_input = torch.cat([tracking_maps] * 2) if do_classifier_free_guidance else tracking_maps
  tracking_maps_input = torch.cat([tracking_maps_input, latents_tracking_image], dim=2)
  ```

**The key insight:** Despite the model being initialized with `in_channels=16`, the patch embedding process in the forward pass handles the 32 channels:

```python
# 2. Patch embedding
hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
hidden_states = self.embedding_dropout(hidden_states)

# Process tracking maps
prompt_embed = encoder_hidden_states.clone()
tracking_maps_hidden_states = self.patch_embed(prompt_embed, tracking_maps)
```

The `patch_embed` layer is designed to handle the combined 32 channels (16 for video + 16 for image conditioning), converting them into the expected embedding dimension for the transformer blocks. Similarly, the tracking path also uses the same `patch_embed` to process its 32 channels.

- Number of transformer blocks: 30 in original model
- Number of tracking blocks: 18 by default (configurable)

Additionally, there's a comment in the unpatchify section that confirms this:
```python
# Note: we use `-1` instead of `channels`:
#   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
#   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
```

### Training Approach
Only specific components are trained during fine-tuning:
- The main transformer blocks remain frozen
- The tracking transformer blocks are trainable
- The "combine_linears" layers that mix tracking features into the main stream are trainable
- The initial_combine_linear is trainable

During training:
1. The model is initialized from the base CogVideoX-5b-I2V model:
   ```python
   transformer = CogVideoXTransformer3DModelTracking.from_pretrained(
       args.pretrained_model_name_or_path,
       subfolder="transformer",
       torch_dtype=load_dtype,
       revision=args.revision,
       variant=args.variant,
       num_tracking_blocks=args.num_tracking_blocks,  # 18 by default
   )
   ```

2. Both the videos and tracking maps are encoded into latents via the VAE:
   ```python
   # Encode videos
   videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
   latent_dist = vae.encode(videos).latent_dist
   video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
   
   # Encode tracking maps
   tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
   tracking_latent_dist = vae.encode(tracking_maps).latent_dist
   tracking_maps = tracking_latent_dist.sample() * VAE_SCALING_FACTOR
   ```

3. The first frames for both are duplicated to create conditioning channels:
   ```python
   # Concatenate [noisy_video_latents, image_latents]  
   noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
   
   # Concatenate [tracking_maps, tracking_image_latents]
   tracking_latents = torch.cat([tracking_maps, tracking_image_latents], dim=2)
   ```
   
4. The model is trained using this architecture - both the main input and tracking input have 8 channels each (4 for content + 4 for conditioning).

This approach allows efficient adaptation without modifying the core generation capabilities of the base model.

The tracking branch takes tracking videos (showing motion/camera paths) and influences the main generation process while preserving spatial relationships.

## Key Files and Directories

### Core Model Architecture
- `/models/cogvideox_tracking.py`: Core implementation of the parallel-stream tracking architecture
  - `CogVideoXTransformer3DModelTracking`: Adds tracking capability to the base transformer
  - The model uses dual transformer blocks - original stream and tracking stream
  - Activation injection mechanism via learnable combine layers
  
### PEFT/LoRA Support
The model includes code to support Parameter-Efficient Fine Tuning (PEFT) through the diffusers library, but doesn't actually use it in the training process:

```python
# From models/cogvideox_tracking.py
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

# In the forward function
if USE_PEFT_BACKEND:
    # weight the lora layers by setting `lora_scale` for each PEFT layer
    scale_lora_layers(self, lora_scale)
else:
    if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
        logger.warning(
            "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
        )
```

Key findings about PEFT/LoRA usage:
1. The training scripts and configurations (`train_DaS.sh`, `args.py`) **do not** have any LoRA-specific configuration
2. The `USE_PEFT_BACKEND` code is inherited from the diffusers library implementation
3. **There is no LoRA-based fine-tuning** - instead, specific components are fully trained:
   ```python
   # From training/cogvideox_image_to_video_sft.py
   text_encoder.requires_grad_(False)
   vae.requires_grad_(False)
   # transformer.requires_grad_(True)  # This is commented out because it sets requires_grad selectively
   ```

4. The actual training approach selectively modifies only certain components:
   ```python
   # In the CogVideoXTransformer3DModelTracking.__init__ method:
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
   ```

So while the codebase maintains compatibility with PEFT, the training uses a direct fine-tuning approach for selected components rather than LoRA adaptation.

### Training
- `/training/cogvideox_image_to_video_sft.py`: Main training script for image-to-video with tracking
- `/training/dataset.py`: Dataset implementation, including `VideoDatasetWithResizingTracking` for tracking maps
- `/scripts/train_DaS.sh`: Training launch script with hyperparameter configurations

### Inference
- `/testing/inference.py`: Core inference logic for generating videos with tracking guidance
- `/models/pipelines.py`: Model pipeline implementation
  - `DiffusionAsShaderPipeline`: High-level pipeline combining all components
  - Tracking visualization and generation functions

### Demo and UI
- `/demo.py`: Command-line demo interface with various generation modes
- `/source/diffusion_as_shader_tester.ipynb`: Interactive notebook for testing
- `/webui.py`: Web interface for the model

### Tracking Tools
- `/models/spatracker/`: Spatial tracker implementation for 3D-aware tracking
- Integrations with external tracking tools:
  - MoGe for monocular geometry estimation
  - VGGT for visual geometry and tracking

## Model Architecture Details

The key innovation in DiffusionAsShader is its parallel dual-stream architecture:

1. **Main Content Stream**: Processes the video/image content and generates the output video
2. **Tracking Stream**: Processes tracking maps (motion, camera movement)
3. **Activation Injection**: After each transformer block, tracking activations are added to the main stream via learnable weights

### Core Forward Function and Tensor Dimensions
The forward function in `CogVideoXTransformer3DModelTracking` is the heart of the architecture:

```python
def forward(
    self,
    hidden_states: torch.Tensor,         # Noise latents + image condition latents
    encoder_hidden_states: torch.Tensor, # Text embeddings from prompt
    tracking_maps: torch.Tensor,         # Tracking latents + tracking image latents
    timestep: Union[int, float, torch.LongTensor],  # Current diffusion timestep
    # ... other parameters ...
):
```

The key sequence of operations for dimensions:

1. **Initial dimensions**:
   ```python
   batch_size, num_frames, channels, height, width = hidden_states.shape
   ```
   - `hidden_states` shape: [batch_size, num_frames, channels, height, width]
   - `encoder_hidden_states` shape: [batch_size, seq_length, embed_dim]

2. **Patch embedding transformation**:
   ```python
   hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
   ```
   - After patch_embed, `hidden_states` includes both text sequence embeddings and patched visual features
   - New shape: [batch_size, text_seq_length + num_patches, embed_dim]
   
3. **Separation of text and image tokens**:
   ```python
   text_seq_length = encoder_hidden_states.shape[1]
   encoder_hidden_states = hidden_states[:, :text_seq_length]  # Text tokens only
   hidden_states = hidden_states[:, text_seq_length:]          # Image tokens only
   ```
   
   This slicing separates:
   - `encoder_hidden_states`: Text tokens [batch_size, text_seq_length, embed_dim]
   - `hidden_states`: Image tokens [batch_size, num_patches, embed_dim]
   
   The same slicing is applied to tracking maps:
   ```python
   tracking_maps = tracking_maps_hidden_states[:, text_seq_length:]
   ```

Note on terminology: The naming of "hidden_states" here is somewhat misleading and is carried over from the diffusers library's API design, where it's a standard parameter name for the main input tensor to transformer blocks. In this specific case:

1. Looking at the original diffusers implementation of CogVideoX:
   ```python
   # From diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py
   latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
   latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
   latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
   
   # This is called with:
   noise_pred = self.transformer(
       hidden_states=latent_model_input,  # The noisy latents being processed
       encoder_hidden_states=prompt_embeds,
       # ...
   )
   ```

2. These aren't actually "hidden" internal states - they're directly the model inputs (noisy latents being denoised)

3. This naming convention likely comes from the transformer architecture's heritage, but is confusing in this specific application

During inference, this function is called from the pipeline's denoising loop:
```python
noise_pred = self.transformer(
    hidden_states=latent_model_input,      # [noisy_video_latents, image_latents]
    encoder_hidden_states=prompt_embeds,   # Text conditioning
    timestep=timesteps,                    # Current timestep
    image_rotary_emb=image_rotary_emb,     # Position encoding
    attention_kwargs=attention_kwargs,
    tracking_maps=tracking_maps_input,     # [tracking_maps, tracking_image_latents]
    return_dict=False,
)[0]
```

Inside the forward function, the key steps are:
1. Process both content and tracking streams through patch embedding
2. Split text tokens and image tokens
3. Run both streams through their respective transformer blocks
4. After each transformer block, mix tracking features into the main stream via learnable weights
5. Final processing and unpatchify to get the output

The model can be initialized in two ways:
- From a DiffusionAsShader checkpoint directly
- From a base CogVideoX model by creating duplicate transformer blocks

### Tracking Maps Generation

Tracking maps visualize correspondence between pixels across frames. Several methods are supported:

1. **SpaTracker** (default): 
   - Used in `batch_tracking.py` and `accelerate_tracking.py`
   - Creates a grid of points and tracks them through the video
   - Visualizes point trajectories on black background (see `models/spatracker/utils/visualizer.py`)
   - Command: `python batch_tracking.py --root /path/to/videos --outdir /path/to/output --grid_size 70`

2. **MoGe**: 
   - Used for single images or when better 3D understanding is needed
   - Creates depth maps and 3D projections
   - Better for camera control from a single image

3. **VGGT/CoTracker**: 
   - Alternative tracking methods for different scenarios
   - VGGT provides camera pose estimation for camera control

Tracking visualization process:
1. Generate a grid of points on the first frame
2. Track these points through all frames of the video
3. Render tracks as colored lines/points on a black background
4. The colors typically represent temporal information (how far from the start frame)

The tracking maps are then encoded through the VAE to produce latents (4 channels) that are injected into the generation process.

## Data Format

For training, the model expects:
- Video files
- Tracking videos showing motion paths
- Text prompts

The dataset structure follows:
```
dataset/
├── prompt.txt
├── videos.txt
├── trackings.txt
├── tracking/
│   ├── 00000_tracking.mp4
│   ├── 00001_tracking.mp4
│   └── ...
└── videos/
    ├── 00000.mp4
    ├── 00001.mp4
    └── ...
```

## Common Commands

### Training
```bash
./scripts/train_DaS.sh
```

### Inference
```bash
python demo.py \
    --prompt "Your prompt text" \
    --checkpoint_path "./diffusion_shader_model" \
    --output_dir "./outputs" \
    --input_path "./input_video.mp4" \
    --tracking_method "spatracker" \
    --gpu 0
```

### Camera Control
```bash
python demo.py \
    --prompt "A car driving down a mountain road" \
    --checkpoint_path "./diffusion_shader_model" \
    --output_dir "./outputs" \
    --input_path "./car.mp4" \
    --camera_motion "trans 0 0 -0.5" \
    --tracking_method "spatracker" \
    --gpu 0
```

### Interactive Testing
Run the notebook at `/source/diffusion_as_shader_tester.ipynb` for interactive testing.