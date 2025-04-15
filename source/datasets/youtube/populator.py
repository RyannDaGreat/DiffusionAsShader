import os
import subprocess
import sys

import fire
import rp

sys.path += rp.get_absolute_paths(
    [
        "~/CleanCode/Management",
        # "~/CleanCode/Github/DiffusionAsShader",
        # "~/CleanCode/Datasets/Vids/Raw_Feb28",
        "~/CleanCode/Github/CogvideX-Interpolation-Mar23:MotionPrompting",
        # "~/CleanCode/Github/CogvideX-Interpolation-Feb13:Inpainting",
    ]
)

from cogvideox_interpolation.datasets.youtube.youtube_dataset import RawYoutubeDataset
import syncutil

def run_diffusion_as_shader_tracker(input_video: str, gpu: int):
    script = r"""
        #!bash
        cd ~/CleanCode/Github/DiffusionAsShader
        export HF_HUB_OFFLINE=1
        python demo.py \
            --prompt "Does not matter" \
            --checkpoint_path diffusion_shader_model \
            --output_dir ${OUTPUT_FOLDER} \
            --input_path "${INPUT_VIDEO}" \
            --tracking_method spatracker \
            --gpu ${GPU} \
            --tracking_only
    """

    input_video = rp.get_absolute_path(input_video)

    output_folder = rp.get_parent_folder(input_video)
    output_folder = rp.path_join(
        output_folder,
        rp.get_file_name(input_video) + "__DiffusionAsShaderCondition",
    )
    output_folder = rp.make_directory(output_folder)

    env = os.environ | {
        "INPUT_VIDEO": input_video,
        "OUTPUT_FOLDER": output_folder,
        "GPU": str(gpu),
    }

    subprocess.run(script, shell=True, env=env)


def run_firstlast_diffusion(sample_path: str, gpu: int, output_filename: str):
    script = r"""
        #!bash
        cd ~/CleanCode/Github/CogvideX-Interpolation-Mar23:MotionPrompting/notebooks
        rp exec '
            device=DEVICE
            exec_ipynb(NOTEBOOK_PATH)
            make_firstlast_mp4(SAMPLE_PATH, OUTPUT_FILENAME)
        ' --DEVICE ${DEVICE} \
          ---NOTEBOOK_PATH "${NOTEBOOK_PATH}" \
          ---SAMPLE_PATH "${SAMPLE_PATH}" \
          ---OUTPUT_FILENAME "${OUTPUT_FILENAME}"
    """

    env = os.environ | {
        "SAMPLE_PATH": sample_path,
        "NOTEBOOK_PATH": "~/CleanCode/Github/CogvideX-Interpolation-Mar23:MotionPrompting/notebooks/FrameInterpTester.ipynb",
        "OUTPUT_FILENAME": output_filename,
        "DEVICE": str(gpu),
    }

    subprocess.run(script, shell=True, env=env)


_setup = False
gpu = None


@rp.globalize_locals
def setup():
    global _setup, gpu
    if _setup:
        return

    dataset = RawYoutubeDataset()

    if gpu is None:
        device = rp.select_torch_device(prefer_used=True, reserve=True)
        gpu = device.index

    _setup = True


@rp.globalize_locals
def process_random_sample():
    setup()

    firstlast_filename = "firstLastInterp_Jack2000.mp4"

    sample = rp.random_element(dataset)
    sample.download()

    rp.pretty_print(sample)

    with rp.SetCurrentDirectoryTemporarily(sample.path):
        run_firstlast_diffusion(sample.path, gpu, firstlast_filename)

        run_diffusion_as_shader_tracker("video.mp4", gpu)
        run_diffusion_as_shader_tracker(firstlast_filename, gpu)

    return sample


@rp.globalize_locals
def main(_gpu: int = None):

    global gpu
    gpu = _gpu

    while True:
        sample = process_random_sample()
        sample.upload()
        # sample.clear_local() #Eh, if we fill 1TB we're doing a good job lol

def launch_all():
    """
    TO LAUNCH:
        bash ~/CleanCode/Management/sync_projects.bash
        python ~/CleanCode/Github/CogvideX-Interpolation-Mar23:MotionPrompting/cogvideox_interpolation/datasets/youtube/populator.py launch_all
    """

    rp.tmux_kill_session('Datagen')

    os.system(
        "rclone sync --progress /home/jupyter/CleanCode/CloudSync/home_cache ~/.cache"
    )  # bash

    syncutil.download(
        "/home/jupyter/CleanCode/Checkpoints/Github/CogvideX-Interpolation-Feb13:Inpainting/checkpoints/randomasks_singleframes_XID=Jack/checkpoint-2000",
        force=True,
    )


    """ python populator.py launch_all """
    yaml=rp.tmuxp_create_session_yaml(
        {f"GPU{i}": f"{sys.executable} {__file__} main {i}" for i in rp.get_all_gpu_ids()},
        session_name="Datagen",
    )
    rp.tmuxp_launch_session_from_yaml(yaml) 

if __name__ == "__main__":
    fire.Fire()
