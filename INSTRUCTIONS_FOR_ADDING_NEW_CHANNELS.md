# Instructions for Adding New Control Channels to DiffusionAsShader

This document provides step-by-step instructions for adding new control channels to the DiffusionAsShader model. The architecture was designed to support multiple control channels, which are used to guide the diffusion process.

## Background

DiffusionAsShader uses a parallel dual-stream architecture:
1. **Main Content Stream**: Processes the video/image content and generates the output video
2. **Control Stream**: Processes tracking maps and other control signals (motion, camera movement, etc.)

The control stream initially supported only one control channel (tracking_maps), but was expanded to support three control channels:
- `tracking_maps`: Primary tracking information
- `counter_tracking_maps`: Secondary tracking information 
- `counter_video_maps`: Video content control information

Each control channel follows the same pattern - they each have their own video data tensor and corresponding first-frame conditioning tensor.

## Channel Dimensions

Each control channel has the same dimensions as the main content stream:
- Each channel has **16 latent dimensions** from the VAE
- Each channel gets concatenated with its first-frame conditioning (also 16 dimensions)
- Total dimensions per channel: **32** (16 for content + 16 for conditioning)
- Current total control dimensions: **96** (32 × 3 channels)

## Files That Need Modification

To add a new control channel, you need to modify these key files:

1. `models/cogvideox_tracking.py`: Core model architecture
2. `training/dataset.py`: Dataset loading and processing
3. `training/args.py`: Command-line arguments
4. `training/cogvideox_image_to_video_sft.py`: Training logic

## Step-by-Step Instructions

### 1. Modify Model Architecture (`models/cogvideox_tracking.py`)

#### 1.1 Update the `second_patch_embed` Initialization

Find the `second_patch_embed` initialization in the `CogVideoXTransformer3DModelTracking.__init__` method and increase the `in_channels` parameter:

```python
self.second_patch_embed = CogVideoXPatchEmbed(
    patch_size=patch_size,
    patch_size_t=None,
    in_channels=in_channels * 4, # UPDATED: For 4 control videos (was 3)
    embed_dim=inner_dim,
    text_embed_dim=text_embed_dim,
    bias=True,
    ...
)
```

#### 1.2 Update the Forward Method Signature

Add your new control channel parameter to the `forward` method:

```python
def forward(
    self,
    hidden_states: torch.Tensor,         # BTCHW
    encoder_hidden_states: torch.Tensor, # B Seq_Len Dim
    
    tracking_maps: torch.Tensor,         # BTCHW
    counter_tracking_maps: torch.Tensor, # BTCHW
    counter_video_maps: torch.Tensor,    # BTCHW
    new_control_maps: torch.Tensor,      # BTCHW - NEW CHANNEL
    
    timestep: Union[int, float, torch.LongTensor],
    ...
)
```

#### 1.3 Update Control Map Concatenation

Add your new control channel to the `control_maps` concatenation:

```python
control_maps = torch.cat(
    [
        tracking_maps,
        counter_tracking_maps,
        counter_video_maps,
        new_control_maps,  # NEW CHANNEL
    ],
    dim=2,
)
```

#### 1.4 Update Shape Validation

Update the shape validation to include the new channel and reflect the increased total dimensions:

```python
rp.validate_tensor_shapes(
    hidden_states         = "B T C H W",
    
    tracking_maps         = "B T C H W",
    counter_tracking_maps = "B T C H W",
    counter_video_maps    = "B T C H W",
    new_control_maps      = "B T C H W",  # NEW CHANNEL
    
    control_maps          = "B T CCCC H W",  # UPDATED: CCCC instead of CCC
    
    encoder_hidden_states = "B Seq Dim",
    
    C=32,
    CCCC=32*4,  # UPDATED: 4 channels instead of 3
    
    verbose=False,
)
```

#### 1.5 Update Model Weight Initialization

If adding compatibility for loading previous model weights, update the initialization in `from_pretrained`:

```python
model.second_patch_embed.proj.weight.data[:,:32,:,:] = model.patch_embed.proj.weight.data
# Rest is automatically zero-initialized
```

### 2. Modify the Dataset (`training/dataset.py`)

#### 2.1 Update `VideoDatasetWithResizingTracking.__init__`

Add your new control channel column parameter:

```python
def __init__(self, *args, **kwargs) -> None:
    self.tracking_column         = kwargs.pop("tracking_column"        , None)
    self.counter_tracking_column = kwargs.pop("counter_tracking_column", None)
    self.counter_video_column    = kwargs.pop("counter_video_column"   , None)
    self.new_control_column      = kwargs.pop("new_control_column"     , None)  # NEW CHANNEL
    
    assert self.tracking_column         is not None
    assert self.counter_tracking_column is not None
    assert self.counter_video_column    is not None
    assert self.new_control_column      is not None  # NEW CHANNEL
    
    super().__init__(*args, **kwargs)
```

#### 2.2 Update the `_load_dataset_from_local_path` Method

Add loading for the new control channel path:

```python
prompt_path = self.data_root.joinpath(self.caption_column)
video_path = self.data_root.joinpath(self.video_column)
tracking_path = self.data_root.joinpath(self.tracking_column)
counter_tracking_path = self.data_root.joinpath(self.counter_tracking_column)
counter_video_path = self.data_root.joinpath(self.counter_video_column)
new_control_path = self.data_root.joinpath(self.new_control_column)  # NEW CHANNEL

# Add verification for the new path
if not new_control_path.exists() or not new_control_path.is_file():
    raise ValueError(
        f"Expected `--new_control_column` to be path to a file in `--data_root` containing line-separated new control information. new_control_path={new_control_path}"
    )

# Add loading for the new path
with open(new_control_path, "r", encoding="utf-8") as file:
    new_control_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

self.new_control_paths = new_control_paths  # Store the paths
```

#### 2.3 Update the `_load_dataset_from_csv` Method

Similarly, add loading for your new control channel from CSV:

```python
new_control_paths = df[self.new_control_column].tolist()
new_control_paths = [self.data_root.joinpath(line.strip()) for line in new_control_paths]
self.new_control_paths = new_control_paths
```

#### 2.4 Update the `_preprocess_video` Method

Add processing for your new control channel:

```python
def _preprocess_video(
    self,
    path: Path,
    tracking_path: Path,
    counter_tracking_path: Path,
    counter_video_path: Path,
    new_control_path: Path,  # NEW CHANNEL
) -> torch.Tensor:
    # Existing loading logic...
    
    # Load new control frames
    new_control_reader = decord.VideoReader(uri=new_control_path.as_posix())
    new_control_frames = new_control_reader.get_batch(frame_indices)
    new_control_frames = new_control_frames[:nearest_frame_bucket].float()
    new_control_frames = new_control_frames.permute(0, 3, 1, 2).contiguous()
    new_control_frames_resized = torch.stack([resize(new_control_frame, nearest_res) for new_control_frame in new_control_frames], dim=0)
    new_control_frames = torch.stack([self.video_transforms(new_control_frame) for new_control_frame in new_control_frames_resized], dim=0)
    
    # Apply same augmentations to new control channel as needed
    if rp.random_chance(.2):
        new_control_frames = new_control_frames.flip(0)
        audit_metadata+=['REV-NEW']
    
    # Return with new channel
    return image, frames, tracking_frames, counter_tracking_frames, counter_video_frames, new_control_frames, None
```

#### 2.5 Update the `__getitem__` Method

Add your new control channel to the return dictionary:

```python
return {
    "prompt": self.id_token + self.prompts[index],
    "image": image,
    "video": video,
    "tracking_map": tracking_map,
    "counter_tracking_map": counter_tracking_map,
    "counter_video_map": counter_video_map,
    "new_control_map": new_control_map,  # NEW CHANNEL
    "video_metadata": {
        "num_frames": video.shape[0],
        "height": video.shape[2],
        "width": video.shape[3],
    },
}
```

### 3. Update Command-Line Arguments (`training/args.py`)

Add new command-line arguments for your control channel:

```python
parser.add_argument(
    "--new_control_column",
    type=str,
    help="The column of the dataset containing the new control map for each video."
)

parser.add_argument(
    "--new_control_map_path",
    type=str,
    help="Path to the new control map video."
)
```

### 4. Update Training Logic (`training/cogvideox_image_to_video_sft.py`)

#### 4.1 Update the `CollateFunctionImageTracking` Class

Add collection of your new control channel:

```python
def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    # Existing collation logic...
    
    new_control_maps = [x["new_control_map"] for x in data]
    new_control_maps = torch.stack(new_control_maps).to(dtype=self.weight_dtype, non_blocking=True)
    
    return {
        "images": images,
        "videos": videos,
        "prompts": prompts,
        "tracking_maps": tracking_maps,
        "counter_tracking_maps": counter_tracking_maps,
        "counter_video_maps": counter_video_maps,
        "new_control_maps": new_control_maps,  # NEW CHANNEL
    }
```

#### 4.2 Update Dataset Initialization

Add your new control channel parameter:

```python
dataset_init_kwargs = {
    "data_root": args.data_root,
    "dataset_file": args.dataset_file,
    "caption_column": args.caption_column,
    "tracking_column": args.tracking_column,
    "counter_tracking_column": args.counter_tracking_column,
    "counter_video_column": args.counter_video_column,
    "new_control_column": args.new_control_column,  # NEW CHANNEL
    "video_column": args.video_column,
    # ... other parameters ...
}
```

#### 4.3 Update VAE Encoding

Add encoding for your new control channel:

```python
# Encode the new control channel
new_control_maps = new_control_maps.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
new_control_latent_dist = cached_vae_encode(new_control_maps)

new_control_image = new_control_image.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
new_control_image_latent_dist = cached_vae_encode(new_control_image)
```

#### 4.4 Update Latent Processing

Add processing for your new control channel:

```python
video_latents          = get_latents_from_dist(latent_dist                  )
tracking_maps          = get_latents_from_dist(tracking_latent_dist         )
counter_tracking_maps  = get_latents_from_dist(counter_tracking_latent_dist )
counter_video_maps     = get_latents_from_dist(counter_video_latent_dist    )
new_control_maps       = get_latents_from_dist(new_control_latent_dist      )  # NEW CHANNEL

# And for the image conditioning
new_control_image_latents = get_image_latents_from_dist(new_control_image_latent_dist, padding_shape)  # NEW CHANNEL
```

#### 4.5 Update Model Input Preparation

Prepare your new control channel for the model:

```python
noisy_model_input        = torch.cat([noisy_video_latents  , image_latents                 ], dim=2)
tracking_latents         = torch.cat([tracking_maps        , tracking_image_latents        ], dim=2)
counter_tracking_latents = torch.cat([counter_tracking_maps, counter_tracking_image_latents], dim=2)
counter_video_latents    = torch.cat([counter_video_maps   , counter_video_image_latents   ], dim=2)
new_control_latents      = torch.cat([new_control_maps     , new_control_image_latents     ], dim=2)  # NEW CHANNEL
```

#### 4.6 Update Model Forward Pass

Pass your new control channel to the model:

```python
model_output = transformer(
    hidden_states=noisy_model_input,
    encoder_hidden_states=prompt_embeds,
    tracking_maps=tracking_latents,
    counter_tracking_maps=counter_tracking_latents,
    counter_video_maps=counter_video_latents,
    new_control_maps=new_control_latents,  # NEW CHANNEL
    timestep=timesteps,
    image_rotary_emb=image_rotary_emb,
    return_dict=False,
)[0]
```

### 5. Update Pipeline in `CogVideoXImageToVideoPipelineTracking`

#### 5.1 Update Pipeline Parameters

Add your new control channel parameters:

```python
def __call__(
    self,
    # ... existing parameters ...
    new_control_maps: torch.Tensor=None,
    new_control_image: torch.Tensor=None,
    # ... other parameters ...
)
```

#### 5.2 Add Assertions for New Parameters

```python
assert tracking_maps is not None
assert tracking_image is not None
assert counter_tracking_maps is not None
assert counter_tracking_image is not None
assert counter_video_maps is not None
assert counter_video_image is not None
assert new_control_maps is not None  # NEW CHANNEL
assert new_control_image is not None  # NEW CHANNEL
```

#### 5.3 Update Image Processing

Add processing for your new control images:

```python
image                  = self.video_processor.preprocess(image                 , height=height, width=width).to(device, dtype=prompt_embeds.dtype)
tracking_image         = self.video_processor.preprocess(tracking_image        , height=height, width=width).to(device, dtype=prompt_embeds.dtype)
counter_tracking_image = self.video_processor.preprocess(counter_tracking_image, height=height, width=width).to(device, dtype=prompt_embeds.dtype)
counter_video_image    = self.video_processor.preprocess(counter_video_image   , height=height, width=width).to(device, dtype=prompt_embeds.dtype)
new_control_image      = self.video_processor.preprocess(new_control_image     , height=height, width=width).to(device, dtype=prompt_embeds.dtype)  # NEW CHANNEL
```

#### 5.4 Update Latent Preparation

Prepare latents for your new control channel:

```python
_, new_control_image_latents = self.prepare_latents(
    new_control_image,
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
```

#### 5.5 Update Denoising Loop

Add your new control channel to the denoising loop:

```python
# Process new control maps
latents_new_control_image = torch.cat([new_control_image_latents] * 2) if do_classifier_free_guidance else new_control_image_latents
new_control_maps_input = torch.cat([new_control_maps] * 2) if do_classifier_free_guidance else new_control_maps
new_control_maps_input = torch.cat([new_control_maps_input, latents_new_control_image], dim=2)

# Update the transformer call
noise_pred = self.transformer(
    hidden_states=latent_model_input,
    encoder_hidden_states=prompt_embeds,
    timestep=timestep,
    image_rotary_emb=image_rotary_emb,
    attention_kwargs=attention_kwargs,
    tracking_maps=tracking_maps_input,
    counter_tracking_maps=counter_tracking_maps_input,
    counter_video_maps=counter_video_maps_input,
    new_control_maps=new_control_maps_input,  # NEW CHANNEL
    return_dict=False,
)[0]
```

## Notes and Recommendations

1. **Backward Compatibility**: To maintain backward compatibility with existing models, modify the model's `from_pretrained` method to handle loading older models without the new control channel.

2. **Data Preparation**: Ensure your new control data follows the same format and dimensions as the existing control channels.

3. **Memory Considerations**: Each additional control channel increases memory usage. Consider optimizing memory usage if adding multiple new channels.

4. **Weight Initialization**: Initialize weights for the new control channel properly, typically by zeroing out the weights to start with a clean slate.

5. **Testing**: After adding a new control channel, thoroughly test both training and inference to ensure the model behaves as expected.

## Debugging Tips

1. **Shape Errors**: If you encounter shape mismatches, double-check that all tensors have the correct dimensions at every processing step.

2. **Memory Errors**: If you encounter out-of-memory errors, consider using gradient checkpointing or reducing batch size.

3. **VAE Caching**: The system uses cached VAE encoding for efficiency. Make sure the cache handles your new control channel correctly.

4. **Path Verification**: Verify that all file paths for your new control channel are correctly formed and accessible.

5. **Dimension Tracing**: Use print statements with tensor shapes at key points to trace dimension changes through the pipeline.

By following these steps, you can successfully add new control channels to the DiffusionAsShader model architecture.