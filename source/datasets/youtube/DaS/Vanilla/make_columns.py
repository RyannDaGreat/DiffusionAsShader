from rp import *

sys.path += rp.get_absolute_paths(
    [
        # "~/CleanCode/Management",
        "~/CleanCode/Github/DiffusionAsShader",
        # "~/CleanCode/Datasets/Vids/Raw_Feb28",
        # "~/CleanCode/Github/CogvideX-Interpolation-Mar23:MotionPrompting",
        # "~/CleanCode/Github/CogvideX-Interpolation-Feb13:Inpainting",
    ]
)

from source.datasets.youtube.youtube_dataset import ProcessedYoutubeDataset

dataset = ProcessedYoutubeDataset()

# Right now just using 1000 samples so I can test the code and make sure it runs
samples = gather(dataset, range(1000))

def prepare_sample(sample):
    if not rp.path_exists(sample.video_480p49_path):
        sample.video_480p49
        sample.upload()

load_files(
    prepare_sample,
    samples,
    show_progress=True,
    strict=True,
    num_threads=100,
)

prompt    = [x.prompt.replace('\n',' ') for x in samples]
videos    = [x.video_480p49_path        for x in samples]
trackings = [x.video_dasTrackvid_path   for x in samples]

prompt_path    = save_file_lines(prompt   , "prompt.txt"   )
videos_path    = save_file_lines(videos   , "videos.txt"   )
trackings_path = save_file_lines(trackings, "trackings.txt")

print(fansi_highlight_path(get_absolute_path(prompt_path   )))
print(fansi_highlight_path(get_absolute_path(videos_path   )))
print(fansi_highlight_path(get_absolute_path(trackings_path)))
