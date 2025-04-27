##########################
# IMPORTS
##########################

import sys
import os
import shlex

from functools import cached_property

import models.cogvideox_tracking as cogtrack
from rp import *
import torch
from icecream import ic

sys.path += rp.get_absolute_paths(
    [
        "~/CleanCode/Management",
        # "~/CleanCode/Github/DiffusionAsShader",
        # "~/CleanCode/Datasets/Vids/Raw_Feb28",
        # "~/CleanCode/Github/CogvideX-Interpolation-Mar23:MotionPrompting",
        # "~/CleanCode/Github/CogvideX-Interpolation-Feb13:Inpainting",
    ]
)

import syncutil


##########################
# FUNCTIONS
##########################

def update_to_latest_checkpoint():
    latest_transformer_checkpoint = checkpoint_root
    
    fansi_print(f'Using checkpoint: {latest_transformer_checkpoint}','bold green undercurl')

    rp.r._run_sys_command(
        "rm",
        "-rf",
        "/home/jupyter/CleanCode/Github/DiffusionAsShader/diffusion_shader_model_CKPT/transformer",
    )
    make_hardlink(
        path_join(latest_transformer_checkpoint, "transformer"),
        "/home/jupyter/CleanCode/Github/DiffusionAsShader/diffusion_shader_model_CKPT/transformer",
        recursive=True,
    )
    

if "pipe" not in vars():
    update_to_latest_checkpoint()
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

    video_path=get_absolute_path(video_path)

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
    return image_form(next(load_video_stream(get_absolute_path(video_path))))

def image_form(image):
    image=as_rgb_image(image)
    return as_pil_image(image)

def run_pipe(
    prompt                    = "A soccer player from Hertha BSC is in the field with the ball while an opposing player is running towards him.",
    video_path                = "~/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/video.mp4",
    tracking_map_path         = "~/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/video.mp4__DiffusionAsShaderCondition/tracking_video.mp4",
    counter_tracking_map_path = "~/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/firstLastInterp_Jack2000.mp4__DiffusionAsShaderCondition/tracking_video.mp4",
    counter_video_map_path    = "~/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/validation_samples/-mYvWIeIEHE_268812917_274856884/firstLastInterp_Jack2000.mp4",
):
    ic(
        prompt,
        video_path,
        tracking_map_path,
        counter_tracking_map_path,
        counter_video_map_path,
    )

    pipeline_args = {
        "prompt"                 : prompt,
        "image"                  : load_video_first_frame(video_path),
        "tracking_image"         : load_video_first_frame(tracking_map_path),
        "counter_tracking_image" : load_video_first_frame(counter_tracking_map_path),
        "counter_video_image"    : load_video_first_frame(counter_video_map_path),
        "tracking_maps"          : get_maps(tracking_map_path),
        "counter_tracking_maps"  : get_maps(counter_tracking_map_path),
        "counter_video_maps"     : get_maps(counter_video_map_path),
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
    return video

@globalize_locals
def run_test(index):
    
    prompts = "/home/jupyter/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/Vanilla/prompt.txt"
    video_paths = "/home/jupyter/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/Vanilla/videos.txt"
    
    prompts = load_file_lines(prompts,use_cache=True)
    video_paths = load_file_lines(video_paths,use_cache=True)

    video_names = get_folder_names(get_paths_parents(video_paths))

    prompts, video_names = list_transpose(unique(list_transpose([prompts, video_names])))

    prompt=prompts[index]
    video_name=video_names[index]
    
    output_root = 'run_test_outputs'
    output_video_path   = f'{output_root}/{video_name}.mp4'
    output_preview_path = f'{output_root}/{video_name}__preview.mp4'
    make_directory(output_root)
    
    root = '/home/jupyter/CleanCode/Datasets/Vids/Raw_Feb28/Processed_April7'
    video_path                = f"{root}/{video_name}/video.mp4"
    tracking_map_path         = f"{root}/{video_name}/video.mp4__DiffusionAsShaderCondition/tracking_video.mp4"
    counter_tracking_map_path = f"{root}/{video_name}/firstLastInterp_Jack2000.mp4__DiffusionAsShaderCondition/tracking_video.mp4"
    counter_video_map_path    = f"{root}/{video_name}/firstLastInterp_Jack2000.mp4"
    
    video_path                = get_absolute_path(video_path               )
    tracking_map_path         = get_absolute_path(tracking_map_path        )
    counter_tracking_map_path = get_absolute_path(counter_tracking_map_path)
    counter_video_map_path    = get_absolute_path(counter_video_map_path   )
    output_video_path         = get_absolute_path(output_video_path        )
    output_preview_path       = get_absolute_path(output_preview_path      )

    video = run_pipe(**gather_vars('prompt video_path tracking_map_path counter_tracking_map_path counter_video_map_path'))
    
    save_video_mp4(video, output_video_path, video_bitrate="max", framerate=15)
    fansi_print(f"Saved Video: {output_video_path}", "bold green italic")

    preview_video = horizontally_concatenated_videos(
        labeled_videos(
            resize_videos_to_min_size(
                resize_lists_to_min_len(
                    load_videos(
                        video_path,
                        output_video_path,
                        counter_video_map_path,
                        tracking_map_path,
                        counter_tracking_map_path,
                    )
                )
            ),
            [
                "Original Video",
                "Output Video",
                "Counter Video",
                "Tracking Video",
                "Counter Tracking Video",
            ],
        )
    )

    save_video_mp4(preview_video, output_preview_path, video_bitrate="max", framerate=15)
    fansi_print(f"Saved Video: {output_preview_path}", "bold green italic")


##########################
# SETTINGS
##########################

checkpoint_root = '/home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans2500100000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-4500'
checkpoint_title = get_folder_name(checkpoint_root)


##########################
# SETUP
##########################

latest_transformer_checkpoint = syncutil.sync_checkpoint_folder(checkpoint_root)

set_current_directory('/home/jupyter/CleanCode/Github/DiffusionAsShader')
if not file_exists('source/datasets/youtube/DaS/Vanilla/prompt.txt'):
    os.system('python source/datasets/youtube/DaS/Vanilla/make_columns.py')
if not folder_exists('diffusion_shader_model_CKPT'):
    make_hardlink('diffusion_shader_model','diffusion_shader_model_CKPT',recursive=True)

device = rp.select_torch_device(prefer_used=True)


##########################
# MAIN
##########################

for index in range(50):
    try:
        run_test(index)
    except Exception:
        print_stack_trace()
