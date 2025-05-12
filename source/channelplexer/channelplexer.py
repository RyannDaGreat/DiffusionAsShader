# 2025-05-05 08:33:17.875913
import torch
import torch.nn.functional as F 
import torch.nn as nn
import rp
import numpy as np
from einops import rearrange
from diffusers import CogVideoXPipeline
import itertools


if not 'pipe' in dir():
    device = rp.select_torch_device(prefer_used=True)
    dtype = torch.float16
    # pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", dtype=dtype)
    pipe = CogVideoXPipeline.from_pretrained("/home/jupyter/CleanCode/Huggingface/CogVideoX-5b-I2V", dtype=dtype)
    pipe.vae=pipe.vae.to(device=device, dtype=dtype)

##################################################

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def ryan_encode_video(
    pipe,
    video,
    *,
    latents_form="BTCHW",
    individual_frames=False,
):
    """
    Encodes a list of PIL images into a video latent representation using the provided VAE and VideoProcessor.

    Args:
        video (List[PIL.Image.Image]): List of PIL images representing the video frames.
        batch_size (int, optional): Batch size for encoding. Defaults to 1.
        num_frames (int, optional): Number of frames in the video. Defaults to 49.
        height (int, optional): Height of each frame. Defaults to 60.
        width (int, optional): Width of each frame. Defaults to 90.
        generator (Optional[torch.Generator], optional): Torch generator for reproducibility. Defaults to None.
        device (str, optional): Device to run encoding on. Defaults to "cuda".

    Returns:
        torch.Tensor: Video latent representation.

        RETURNS IN FORM: BTCHW  (the default)
            OR other forms: TCHW, CHW
    """
    # COGVIDEOX ASSUMPTIONS
    height = rp.get_video_height(video)
    width = rp.get_video_width(video)
    num_frames = len(video)
    assert num_frames <= 49, num_frames  # Warning! Not strict error though.

    if individual_frames:
        #Encode each frame individually. Used for original first-frame behaviour.
        video_latents = [
            ryan_encode_video(
                pipe,
                [frame],
                latents_form="BTCHW",
                individual_frames=False,
            )
            for frame in video
        ]
        video_latents = torch.cat(video_latents, dim=1)

    else:
        video = rp.as_rgb_images(video)
        video = rp.as_pil_images(video)
        vae = pipe.vae
        video_processor = pipe.video_processor

        video_tensor = video_processor.pil_to_numpy(video)
        video_tensor = video_processor.numpy_to_pt(video_tensor)
        video_tensor = video_tensor.to(device=vae.device, dtype=vae.dtype)

        # Resize and crop images to match model input size
        processed_video = []
        for frame in video_tensor:
            frame = video_processor.preprocess(frame, height=height, width=width)
            processed_video.append(frame)
        processed_video = torch.concatenate(processed_video)  # [F, C, H, W]
        processed_video = processed_video.unsqueeze(0)  # [B=1, F, C, H, W]

        processed_video = rearrange(processed_video, "B F C H W -> B C F H W")

        with torch.no_grad():
            video_latents = vae.encode(processed_video)
            
        video_latents = retrieve_latents(video_latents)
        video_latents = [video_latents]  # Encode the entire video at once

        video_latents = (
            torch.cat(video_latents, dim=0).to(vae.dtype).permute(0, 2, 1, 3, 4)
        )  # [B, C, F, H, W]
        # <---- THIS COMMENT IS WRONG!! DISCOVERED BY CLARA. IT'S ACTUALLY BTCHW

        if not vae.config.invert_scale_latents:
            video_latents = vae.config.scaling_factor * video_latents
        else:
            video_latents = 1 / vae.config.scaling_factor * video_latents

    if latents_form == "BTCHW":
        return video_latents
    elif latents_form == "TCHW":
        return rearrange(video_latents, "1 T C H W -> T C H W")
    elif latents_form == "CHW":
        assert num_frames==1, num_frames
        return rearrange(video_latents, "1 1 C H W -> C H W")
    else:
        assert False, latents_form

def ryan_decode_latents(
    pipe,
    latents,
    *,
    latents_form=None,
):
    """Returns a video"""
    if latents_form is None: latents_form = {5:'BTCHW', 4:'TCHW', 3:'CHW'}[latents.ndim]
    if latents_form == "BTCHW":
        pass
    elif latents_form == "TCHW":
        latents = rearrange(latents, "T C H W -> 1 T C H W") #BTCHW
    elif latents_form == "CHW":
        latents = rearrange(latents,   "C H W -> 1 1 C H W") #BTCHW
    else:
        assert False, latents_form

    with torch.no_grad():
        video = pipe.decode_latents(latents)
    
    video = rearrange(video, "1 C T H W -> T C H W")
    video = video / 2 + 0.5
    video = rp.as_numpy_images(video)
    
    return video

def encode(video, *, individual_frames=False):
    video = rp.as_numpy_images(video)
    return ryan_encode_video(
        pipe,
        video,
        individual_frames=individual_frames,
        latents_form="TCHW",
    )

def decode(latents):
    return ryan_decode_latents(pipe, latents.to(device=device, dtype=dtype))


##################################################

def random_color_video(VT, VH, VW, VC):
    colors = np.random.rand(VT, 1, 1, VC)
    output = np.tile(colors, (1, VH, VW, 1))

    output = output / 2 + np.roll(output, 1, 0) / 2
    # output = output / 2 + np.roll(output, 1, 0) / 2
    # output = output / 2 + np.roll(output, 1, 0) / 2
    output = rp.full_range(output)
    
    assert output.shape == (VT, VH, VW, VC)
    return output
    
if rp.running_in_jupyter_notebook():
    rp.display_image_slideshow(rp.as_numpy_images(random_color_video(49,128,128,3)))

##################################################

#video_urls = 'https://gist.githubusercontent.com/SqrtRyan/ef34afaa17a92503ea5f2928780b4497/raw/26e97b6a5d70b4464c1fd967cff9132a62cb1c12/gistfile1.txt'
#video_urls = rp.download_to_cache(video_urls)
#video_urls = rp.load_file_lines(video_urls, use_cache=True)
#video_urls = video_urls[:1000]
#
#video_paths = rp.download_files_to_cache(video_urls, show_progress=True)

video_paths = load_file_lines('/home/jupyter/CleanCode/Github/DiffusionAsShader/source/datasets/youtube/DaS/Vanilla/videos.txt')

train_videos = rp.load_videos(video_paths, use_cache=True, show_progress=True, lazy=False, num_threads=100)

def random_color_video(VT, VH, VW, VC):
    assert VC==3

    video = rp.random_element(train_videos)
    video = rp.resize_list(video, VT)
    video = rp.as_float_images(video)
    video = rp.as_rgb_images(video)
    video = rp.resize_images(video, size=(VH, VW))

    output = video
    
    assert output.shape == (VT, VH, VW, VC)
    return output
    
if rp.running_in_jupyter_notebook():
    rp.display_image_slideshow(rp.as_numpy_images(random_color_video(49,128,128,3)))

##################################################

# VH = VW = 128
VH, VW = 60*8, 90*8

VT, VH, VW, VC = 49, VH, VW, 3
LT, LH, LW, LC = 13, VH//8, VW//8, 16

TI = VT
TO = LT
CI = CO = LC

TICI = TI * CI
TOCO = TO * CO
LHLW = LH * LW

##################################################

def get_sample(video=None):
    if video is None: video = random_color_video(VT, VH, VW, VC)

    #Make sure they're small and fast...
    # video = rp.resize_images(video, size=(VH, VW)) 
    
    latents = encode(video, individual_frames=False)

    ilatents = encode(video, individual_frames=True) #Individual Latents

    latents = latents.to('cpu')
    ilatents = ilatents.to('cpu')
    
    return rp.gather_vars('video latents ilatents')

##################################################

@rp.globalize_locals
def demo_get_sample():
    sample = get_sample()
    video, latents, ilatents = rp.destructure(sample)
    pred_video = decode(latents)
    preview_video = rp.horizontally_concatenated_videos(rp.as_numpy_videos(rp.resize_lists_to_max_len([video, pred_video])))
    rp.display_image_slideshow(preview_video)
    rp.display_image(rp.horizontally_concatenated_images(rp.rotate_images(preview_video,angle=90)))

if rp.running_in_jupyter_notebook():
    demo_get_sample()

##################################################

sample_folder = f'~/pcached_samples_videos_{VH}_{VW}'
rp.fansi_print(sample_folder, 'bold white blue cyan')
num_samples = 15
sample_paths = rp.path_join(sample_folder, list(map(str,range(num_samples))))
samples = [rp.file_cache_call(path, get_sample) for path in rp.eta(sample_paths, 'Making Samples')]

##################################################

def f(LI, W):
    # assert LI.shape == (TI, CI, LH, LW)
    # assert W.shape == (TICI, TOCO)
    LI = rearrange(LI, 'TI CI LH LW -> (LH LW) (TI CI)')
    # assert LI.shape == (LHLW, TICI)
    LO = LI @ W
    # assert LO.shape == (LHLW, TOCO)
    LO = rearrange(LO, '(LH LW) (TO CO) -> TO CO LH LW', TO=TO, CO=CO, LH=LH, LW=LW)
    # assert LO.shape == (TO, CO, LH, LW)
    return LO

##################################################

W_mask = torch.eye(LC, device=device, dtype=dtype).repeat((TI,TO))
W_mask[:]=1 #DISABLE THE MASK
rp.display_image(rp.as_rgb_image(rp.as_numpy_array(W_mask)))

##################################################

W = torch.zeros(TICI, TOCO, dtype=dtype, device=device, requires_grad=True)
# W = nn.Parameter(W)
optim = torch.optim.SGD(params=[W], lr=.01, momentum=.9)

##################################################

train_iters = 10000
losses = []
batch_size = 10
for train_iter in range(1, train_iters+1):
    loss = 0
    for _ in range(batch_size):
        # W = W * W_mask
        
        sample = rp.random_element(samples)
        L  = sample.latents.to(device) + 0
        LI = sample.ilatents.to(device) + 0
        LO = f(LI, W * W_mask)
        assert L.shape == LO.shape
        
        loss = loss + F.mse_loss(L, LO)

    losses.append(float(loss))
    
    loss.backward()
    optim.step()
    optim.zero_grad()

    if rp.toc() > 1 or train_iter==train_iters:
        print(f'{train_iter: >7}    {float(loss):.03}')
        rp.tic()
    

##################################################

@rp.globalize_locals
def demo_get_sample():
    sample = get_sample()
    video, latents, ilatents = rp.destructure(sample)
    latents = latents.to(device)
    ilatents = ilatents.to(device)
    pred_video = decode(latents)
    f_pred_video = decode(f(ilatents, W))
    preview_video = rp.horizontally_concatenated_videos(rp.as_numpy_videos(rp.resize_lists_to_max_len([video, pred_video, f_pred_video])))
    rp.display_image_slideshow(preview_video)
    rp.display_image(rp.horizontally_concatenated_images(rp.rotate_images(preview_video,angle=90)))
demo_get_sample()

##################################################

@rp.globalize_locals
def load_in_video(url):
    in_video = url
    # in_video = 'https://videos.pexels.com/video-files/2795691/2795691-uhd_2560_1440_25fps.mp4'
    # in_video = 'https://videos.pexels.com/video-files/6507082/6507082-hd_1920_1080_25fps.mp4'
    # in_video = 'https://videos.pexels.com/video-files/6507468/6507468-hd_1920_1080_25fps.mp4'
    in_video = rp.download_to_cache(in_video)
    in_video = rp.load_video(in_video, use_cache=True)
    in_video = rp.resize_list(in_video, VT)
    in_video = rp.resize_images(in_video, size=(VH, VW))
    rp.display_video(in_video, framerate=10)
    

##################################################

# with rp.TemporarilySetItem(globals(), dict(VH = 64, VW = 64, LW=8, LH=8)):
@rp.globalize_locals
def demo_get_sample():
    global video
    sample = get_sample(in_video[::-1])
    video, latents, ilatents = rp.destructure(sample)
    latents = latents.to(device)
    ilatents = ilatents.to(device)
    pred_video = decode(latents)
    f_pred_video = decode(f(ilatents, W))
    preview_video = rp.vertically_concatenated_videos(rp.as_numpy_videos(rp.resize_lists_to_max_len([video, pred_video, f_pred_video])))
    preview_video = rp.labeled_images(preview_video, range(1,len(preview_video)+1), font='R:Futura')
    # rp.display_image_slideshow(preview_video)
    # rp.display_image(rp.horizontally_concatenated_images(preview_video))
    rp.display_video(rp.resize_images(preview_video,size=.25))

for url in [
    'https://videos.pexels.com/video-files/7515906/7515906-hd_1920_1080_30fps.mp4'
    # 'https://videos.pexels.com/video-files/3127017/3127017-uhd_2560_1440_24fps.mp4',
    # 'https://videos.pexels.com/video-files/1851190/1851190-uhd_2560_1440_25fps.mp4',
    # 'https://videos.pexels.com/video-files/857195/857195-hd_1280_720_25fps.mp4',
    # 'https://videos.pexels.com/video-files/4061791/4061791-hd_1920_1080_24fps.mp4',
    # 'https://videos.pexels.com/video-files/7170786/7170786-uhd_2732_1440_25fps.mp4',
]:
    load_in_video(url)
    demo_get_sample()
    rp.save_video_mp4(preview_video, rp.get_unique_copy_path('linear_combinations_interp.mp4'))

##################################################

rp.display_video(preview_video)

##################################################

rp.save_video_mp4(preview_video, rp.get_unique_copy_path('linear_combinations_interp.mp4'))

##################################################

rp.display_image(np.abs(rp.as_rgb_image(rp.as_numpy_array(W))))
rp.display_image(rp.full_range(np.log(.01+np.abs(rp.as_rgb_image(rp.as_numpy_array(W))))))
# rp.display_image(rp.as_rgb_image(rp.as_numpy_array(rearrange(W, '(TI CI) (TO CO) -> (TI TO) (CI CO)', TO=TO, CO=CO, TI=TI, CI=CI ))))
# rp.display_image(rp.as_rgb_image(rp.as_numpy_array(rearrange(W, '(TI CI) (TO CO) -> (TO TI) (CO CI)', TO=TO, CO=CO, TI=TI, CI=CI ))))

rp.display_image(rp.full_range(rp.cv_resize_image(rp.cv_resize_image(np.abs(rp.as_rgb_image(rp.as_numpy_array(W))), 1/16),16,interp='nearest')))

##################################################

rp.object_to_file(W,'W')

##################################################

rp.display_video(
    rp.horizontally_concatenated_videos(rp.resize_lists_to_max_len(video,
    rp.resize_images(
        rp.full_range(
            rp.as_numpy_images(
                latents[:, :3],
            ),
        ),
        size=(480, 720),
        interp="nearest",
    ))),framerate=20,
)
