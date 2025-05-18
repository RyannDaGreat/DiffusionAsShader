##########################
# IMPORTS
##########################

import sys
import os
import shlex

from functools import cached_property

import models.cogvideox_tracking as cogtrack
import rp
import torch
from icecream import ic

import numpy as np

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

device = rp.select_torch_device(prefer_used=True, reserve=True)

##########################
# FUNCTIONS
##########################

def update_to_latest_checkpoint():
    latest_transformer_checkpoint = checkpoint_root
    
    rp.fansi_print(f'Using checkpoint: {latest_transformer_checkpoint}','bold green undercurl')

    rp.r._run_sys_command(
        "rm",
        "-rf",
        "/home/jupyter/CleanCode/Github/DiffusionAsShader/diffusion_shader_model_CKPT/transformer",
    )
    rp.make_hardlink(
        rp.path_join(latest_transformer_checkpoint, "transformer"),
        "/home/jupyter/CleanCode/Github/DiffusionAsShader/diffusion_shader_model_CKPT/transformer",
        recursive=True,
    )
    

def get_maps(video_path):
    from diffusers.utils import export_to_video, load_image, load_video

    video_path=rp.get_absolute_path(video_path)

    maps = load_video(video_path)
    # Convert list of PIL Images to tensor [T, C, H, W]
    maps = torch.stack(
        [
            torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
            for frame in maps
        ]
    )
    maps = maps.to(device=device, dtype=torch.bfloat16)

    print(f"Encoding tracking maps from {video_path}")
    maps = maps.unsqueeze(0)  # [B, T, C, H, W]
    maps = maps.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    
    maps = maps * 2 - 1 #Normalize from [0,1] to [-1, 1]
    
    with torch.no_grad():
        latent_dist = pipe.vae.encode(maps).latent_dist
        maps = latent_dist.sample() * pipe.vae.config.scaling_factor
        maps = maps.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    
    return maps

def load_video_first_frame(video_path):
    return image_form(next(rp.load_video_stream(rp.get_absolute_path(video_path))))

def image_form(image):
    image=rp.as_rgb_image(image)
    return rp.as_pil_image(image)

@rp.globalize_locals
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

    # prompt = ''
    # fansi_print("LOOK MA NO PROMPT",'blue')

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
        "guidance_scale"      : 3,
        "num_inference_steps" : 30,
    }

    pipeline_args |= dict(          
        use_image_conditioning=True,
        # latent_conditioning_dropout=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # latent_conditioning_dropout=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #Weird, not as good actually...
        latent_conditioning_dropout=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], #Sparse...25%

        # use_image_conditioning=False,
        # latent_conditioning_dropout=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], #Weird, not as good actually...
        # latent_conditioning_dropout=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        # latent_conditioning_dropout=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #latent_conditioning_dropout=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )

    pipeline_args = rp.as_easydict(pipeline_args)

    with torch.no_grad():
        results=pipe(**pipeline_args)
    
    video=results.frames[0]
    video=rp.as_numpy_images(video)
    video = rp.labeled_images(
        video,
        f"PROMPT={repr(prompt[:50])}\nCFG={pipeline_args.guidance_scale} DYN-CFG={pipeline_args.use_dynamic_cfg} STEPS={pipeline_args.num_inference_steps} {''.join(map(str,pipeline_args['latent_conditioning_dropout']))}",
        size=-25,
        background_color="translucent dark blue",
        size_by_lines=False,
    )
    return video

@rp.globalize_locals
def run_test(index):
    
    prompts     = "/home/jupyter/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/Vanilla/prompt.txt"
    video_paths = "/home/jupyter/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/Vanilla/videos.txt"
    
    prompts     = rp.load_file_lines(prompts,use_cache=True)
    video_paths = rp.load_file_lines(video_paths,use_cache=True)

    video_names = rp.get_folder_names(rp.get_paths_parents(video_paths))

    prompts, video_names = rp.list_transpose(rp.unique(rp.list_transpose([prompts, video_names])))

    prompt=prompts[index]
    video_name=video_names[index]
    
    output_root = 'run_test_outputs'
    output_video_path   = f'{output_root}/{video_name}.mp4'
    output_preview_path = f'{output_root}/{video_name}__preview.mp4'
    rp.make_directory(output_root)
    
    root = '/home/jupyter/CleanCode/Datasets/Vids/Raw_Feb28/Processed_April7'
    video_path                = f"{root}/{video_name}/video.mp4"
    tracking_map_path         = f"{root}/{video_name}/video.mp4__DiffusionAsShaderCondition/tracking_video.mp4"
    counter_tracking_map_path = f"{root}/{video_name}/firstLastInterp_Jack2000.mp4__DiffusionAsShaderCondition/tracking_video.mp4"
    counter_video_map_path    = f"{root}/{video_name}/firstLastInterp_Jack2000.mp4"
    
    video_path                = rp.get_absolute_path(video_path               )
    tracking_map_path         = rp.get_absolute_path(tracking_map_path        )
    counter_tracking_map_path = rp.get_absolute_path(counter_tracking_map_path)
    counter_video_map_path    = rp.get_absolute_path(counter_video_map_path   )
    output_video_path         = rp.get_absolute_path(output_video_path        )
    output_preview_path       = rp.get_absolute_path(output_preview_path      )
    
    def make_480p49(video_path):
        new_video_path = video_path+'_480p49.mp4'
        video=rp.load_video(video_path)

        if video.shape==(49,480,720,3):
            return video_path

        video=rp.resize_list(video,49)
        video=rp.resize_images(video,size=(480,720))
        print('make_480p49: NEW VIDEO SHAPE:',video.shape)
        rp.save_video_mp4(video,new_video_path,video_bitrate='max')
        return new_video_path
    video_path = make_480p49(video_path)

    if 1:
        #Use real one as input...
        counter_tracking_map_path, tracking_map_path = tracking_map_path, counter_tracking_map_path
        video_path, counter_video_map_path = counter_video_map_path, video_path

    if 1:

        #Try to slow it down, for testing...
        def make_halfspeed(video_path):
            halfspeed_video_path = video_path+'_halfspeed.mp4'
            #if file_exists(halfspeed_video_path):
                #return halfspeed_video_path
            video=rp.load_video(video_path)
            video=rp.resize_list(video,len(video)*2)[:len(video)]
            print('make_halfspeed: NEW VIDEO SHAPE:',video.shape)
            rp.save_video_mp4(video,halfspeed_video_path,video_bitrate='max')
            return halfspeed_video_path

        #Try to reverse it, for testing...
        def make_reversed(video_path):
            halfspeed_video_path = video_path+'_reversed.mp4'
            video=rp.load_video(video_path)
            video=video[::-1]
            print('make_reversed: NEW VIDEO SHAPE:',video.shape)
            rp.save_video_mp4(video,halfspeed_video_path,video_bitrate='max')
            return halfspeed_video_path
        
        
        video_path=make_halfspeed(video_path)
        tracking_map_path=make_halfspeed(tracking_map_path)
        output_video_path+='_halfspeed.mp4'
        output_preview_path+='_halfspeed.mp4'

        # counter_video_map_path = make_reversed(video_path)
        # counter_tracking_map_path = make_reversed(tracking_map_path)
        # output_video_path+='_reverseOrig.mp4'
        # output_preview_path+='_reverseOrig.mp4'

        # counter_video_map_path = make_reversed(counter_video_map_path)
        # counter_tracking_map_path = make_reversed(counter_tracking_map_path)
        # output_video_path+='_reverse.mp4'
        # output_preview_path+='_reverse.mp4'


    video = run_pipe(**rp.gather_vars('prompt video_path tracking_map_path counter_tracking_map_path counter_video_map_path'))
    
    rp.save_video_mp4(video, output_video_path, video_bitrate="max", framerate=15)
    rp.fansi_print(f"Saved Video: {output_video_path}", "bold green italic")

    preview_video = rp.horizontally_concatenated_videos(
        rp.labeled_videos(
            rp.resize_videos_to_min_size(
                rp.resize_lists_to_min_len(
                    rp.load_videos(
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

    rp.save_video_mp4(preview_video, output_preview_path, video_bitrate="max", framerate=15)
    rp.fansi_print(f"Saved Video: {output_preview_path}", "bold green italic")


##########################
# SETTINGS
##########################

checkpoint_root = '/home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans2500100000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-4500'
checkpoint_root = '/home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans_RandomSpeed_2500_10000000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-1100'
checkpoint_root = '/home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans_RandomSpeed_2500_10000000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-6000'
checkpoint_root = '/home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans_RandomSpeed_WithDropout_2500_10000000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-3000'
checkpoint_root = '/home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans_RandomSpeed_WithDropout_2500_10000000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-9200'
checkpoint_root = '/home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans_RandomSpeed_WithDropout_2500_10000000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-14700'
checkpoint_root = '/home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans_RandomSpeed_WithDropout_2500_10000000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-29000'
checkpoint_root = '/home/jupyter/CleanCode/Github/DiffusionAsShader/ckpts/your_ckpt_path/CounterChans_RandomSpeed_WithDropout_2500_10000000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-29000'

checkpoint_title = rp.get_folder_name(checkpoint_root)

USE_T2V=True
# USE_T2V=False

if USE_T2V:
    os.environ['T2V_TRANSFORMER_CHECKPOINT'] = "/home/jupyter/CleanCode/Huggingface/CogVideoX-5b/transformer"

NO_CONTROLNET=False
if NO_CONTROLNET:
    os.environ['DISABLE_CONTROLNET'] = "True"


##########################
# SETUP
##########################
    
latest_transformer_checkpoint = syncutil.sync_checkpoint_folder(checkpoint_root)

rp.set_current_directory('/home/jupyter/CleanCode/Github/DiffusionAsShader')
if not rp.file_exists('source/datasets/youtube/DaS/Vanilla/prompt.txt'):
    rp.r._run_sys_command('python source/datasets/youtube/DaS/Vanilla/make_columns.py')
if not rp.folder_exists('diffusion_shader_model_CKPT'):
    rp.make_hardlink('diffusion_shader_model','diffusion_shader_model_CKPT',recursive=True)

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

##########################
# MAIN
##########################

num_videos = 200
index_offset = int(device.index) * num_videos // 8
for index in range(num_videos):
    index += index_offset
    index %= num_videos

    try:
        run_test(index)
    except Exception:
        rp.print_stack_trace()
