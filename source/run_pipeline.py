import models.cogvideox_tracking as cogtrack
from rp import *
import torch

validation_prompt         = "A soccer player from Hertha BSC is in the field with the ball while an opposing player is running towards him."
tracking_map_path         = "~/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/video.mp4__DiffusionAsShaderCondition/tracking_video.mp4"
counter_tracking_map_path = "~/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/firstLastInterp_Jack2000.mp4__DiffusionAsShaderCondition/tracking_video.mp4"
counter_video_map_path    = "~/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/firstLastInterp_Jack2000.mp4"
first_frame_image_path    = "~/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/video_firstFrame.png"

first_frame_image_path    = rp.get_absolute_path(first_frame_image_path)
tracking_map_path         = rp.get_absolute_path(tracking_map_path)
counter_tracking_map_path = rp.get_absolute_path(counter_tracking_map_path)
counter_video_map_path    = rp.get_absolute_path(counter_video_map_path)

device = rp.select_torch_device(prefer_used=True)

if "pipe" not in vars():
    pipe = cogtrack.CogVideoXImageToVideoPipelineTracking.from_pretrained(
        "/home/jupyter/CleanCode/Github/DiffusionAsShader/diffusion_shader_model_CKPT"
    )

pipe.to(dtype=torch.bfloat16)
pipe.to(device)
#pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.transformer.eval()
pipe.text_encoder.eval()
pipe.vae.eval()

def get_maps(video_path):
    from diffusers.utils import export_to_video, load_image, load_video

    maps = load_video(video_path)
    # Convert list of PIL Images to tensor [T, C, H, W]
    maps = torch.stack(
        [
            torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
            for frame in maps
        ]
    )
    maps = maps.to(device=device, dtype=torch.bfloat16)
    first_frame = maps[0:1]  # Get first frame as [1, C, H, W]
    height, width = first_frame.shape[2], first_frame.shape[3]

    print(f"Encoding tracking maps from {video_path}")
    maps = maps.unsqueeze(0)  # [B, T, C, H, W]
    maps = maps.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    with torch.no_grad():
        latent_dist = pipe.vae.encode(maps).latent_dist
        maps = latent_dist.sample() * pipe.vae.config.scaling_factor
        maps = maps.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    
    return maps

def load_video_first_frame(video_path):
    return image_form(next(load_video_stream(video_path)))

def image_form(image):
    image=as_rgb_image(image)
    return as_pil_image(image)
    #image=cv_resize_image(image,size=(480,720))
    #image=as_torch_image(image)
    #image=image.to(device=device)
    #image=image.to(dtype=torch.bfloat16)
    #image=image[None,:,None]#CHW -> BCFHW aka 1C1HW
    #return image

pipeline_args = {
    "image"                  : image_form(load_image(first_frame_image_path)),
    "prompt"                 : validation_prompt,
    "tracking_maps"          : get_maps(tracking_map_path),
    "counter_tracking_maps"  : get_maps(counter_tracking_map_path),
    "counter_video_maps"     : get_maps(counter_video_map_path),
    "tracking_image"         : load_video_first_frame(tracking_map_path),
    "counter_tracking_image" : load_video_first_frame(counter_tracking_map_path),
    "counter_video_image"    : load_video_first_frame(counter_video_map_path),
    "negative_prompt"        : "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
    "height"              : 480,
    "width"               : 720,
    "num_frames"          : 49,
    "use_dynamic_cfg"     : True,
    "guidance_scale"      : 6,
    "num_inference_steps" : 30,
}



with torch.no_grad():
    results=pipe(**pipeline_args)
    video=results.frames[0]