import rp
import sys

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
samples = rp.gather(dataset, range(2500))
# samples = rp.gather(dataset, range(10))

def prepare_sample(sample):
    if not rp.path_exists(sample.video_480p49_path):
        sample.video_480p49
        sample.upload()

rp.load_files(
    prepare_sample,
    samples,
    show_progress=True,
    strict=True,
    num_threads=100,
)

root = rp.get_path_parent(__file__)

prompt            = [x.prompt.replace('\n',' ')          for x in samples]
videos            = [x.video_480p49_path                 for x in samples]
trackings         = [x.video_dasTrackvid_path            for x in samples]
counter_trackings = [x.cogxCounterVideo_dasTrackvid_path for x in samples]
counter_videos    = [x.cogxCounterVideo_path             for x in samples]

prompt_path            = rp.save_file_lines(prompt           , rp.path_join(root, "prompt.txt"           ))
videos_path            = rp.save_file_lines(videos           , rp.path_join(root, "videos.txt"           ))
trackings_path         = rp.save_file_lines(trackings        , rp.path_join(root, "trackings.txt"        ))
counter_trackings_path = rp.save_file_lines(counter_trackings, rp.path_join(root, "counter_trackings.txt"))
counter_videos_path    = rp.save_file_lines(counter_videos   , rp.path_join(root, "counter_videos.txt"   ))

print(rp.fansi_highlight_path(rp.get_absolute_path(prompt_path           )))
print(rp.fansi_highlight_path(rp.get_absolute_path(videos_path           )))
print(rp.fansi_highlight_path(rp.get_absolute_path(trackings_path        )))
print(rp.fansi_highlight_path(rp.get_absolute_path(counter_trackings_path)))
print(rp.fansi_highlight_path(rp.get_absolute_path(counter_videos_path   )))

