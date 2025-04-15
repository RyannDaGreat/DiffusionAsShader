# 2025-04-07 23:30:58.046646
# XCloud Common Import Paths
import rp
import sys
import os
import shlex

from functools import cached_property

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

##############################

here = rp.get_parent_folder(__file__)


@rp.memoized
@rp.file_cache_wrap(
    rp.path_join(here, ".cache/vids_gsls.txt"),
    rp.save_file_lines,
    rp.load_file_lines,
)
def youtube_gsls():
    # To refresh, run youtube_gsls.clear_cache()
    return syncutil.gsutil_ls("~/CleanCode/Datasets/Vids/Raw_Feb28/vids")


@rp.memoized
@rp.file_cache_wrap(
    "~/CleanCode/Datasets/Vids/Raw_Feb28/Processed_April7__gsls-cache.txt",
    rp.save_file_lines,
    rp.load_file_lines,
)
def processed_youtube_gsls():
    # To refresh, run processed_youtube_gsls.clear_cache()
    return syncutil.gsutil_ls("~/CleanCode/Datasets/Vids/Raw_Feb28/Processed_April7")


@rp.memoized
def youtube_gs_pairs():
    youtube_gs_pairs = rp.cluster_by_key(
        youtube_gsls(),
        key=lambda url: url.replace(".mp4", "").replace("_text.txt", ""),
    )
    youtube_gs_pairs = [sorted(pair) for pair in youtube_gs_pairs if len(pair) == 2]
    # EXAMPLE: [
    #              ...,
    #              [
    #                  'gs://xcloud-shared/burgert/CleanCode/Datasets/Vids/Raw_Feb28/vids/srl24IxoHSE_294941855_300943717.mp4',
    #                  'gs://xcloud-shared/burgert/CleanCode/Datasets/Vids/Raw_Feb28/vids/srl24IxoHSE_294941855_300943717_text.txt'
    #              ],
    #              ...,
    #          ]
    return youtube_gs_pairs


class GsSample:
    def __init__(self, loc=None):
        self.url = syncutil.get_xcloud_url(loc)
        self.path = syncutil.get_local_cleancode_path(loc)
        self._load_cache={}

    def download(self):
        syncutil.download(self.url, force=True)
        self._downloaded=True
        return self

    def upload(self):
        syncutil.upload(self.path, force=True)
        return self

    def delete_local(self):
        os.system("rm -rf " + shlex.quote(self.path))
        return self

    def release(self):
        self.upload()
        self.delete_local()
        return self
    
    def path_join(self, file):
        return rp.path_join(self.path, file)
    
    def load_file(self, file):
        load = rp.r._omni_load

        path = self.path_join(file)

        if not rp.path_exists(path):
            self.download()

        if path not in self._load_cache:
            self._load_cache[path] = load(path)
            
        return self._load_cache[path]

    def __repr__(self):
        return f"GsSample(path={self.path}, url={self.url})"
    
    def _create_properties(self, properties):
        """Helper to create cached properties with consistent naming pattern
        
        Args:
            properties: Dict mapping property names to file paths

        Creates properties like video, video_path, prompt, prompt_path etc
        """
        for name, path in properties.items():
            # Create dynamic property getter functions
            def make_getter(path_value):
                return lambda self: self.load_file(path_value)
                
            def make_path_getter(path_value):
                return lambda self: self.path_join(path_value)
                
            # Define the properties with proper descriptors
            setattr(self.__class__, name, property(make_getter(path)))
            setattr(self.__class__, f"{name}_path", property(make_path_getter(path)))

class RawYoutubeGsSample(GsSample):
    ROOT = rp.get_absolute_path("~/CleanCode/Datasets/Vids/Raw_Feb28/Processed_April7")

    def __init__(self, sample_name, video_url, prompt_url):
        self.sample_name = sample_name
        self.video_url = video_url
        self.prompt_url = prompt_url

        loc = rp.path_join(self.ROOT, sample_name)
        super().__init__(loc)

        self.video_path = rp.path_join(loc, "video.mp4")
        self.prompt_path = rp.path_join(loc, "prompt.txt")

    def download(self):
        rp.make_directory(self.path)

        rp.par_map(
            rp.download_url,
            [self.video_url, self.prompt_url],
            [self.video_path, self.prompt_path],
        )

        return self.path


class ProcessedYoutubeGsSample(GsSample):
    """ 
    Sample containing YouTube processed data 
    PROPERTIES:
        .path
        .video
        .video_path
        .cogxCounterVideo
        .cogxCounterVideo_path
        .video_dasTrackvid
        .video_dasTrackvid_path
        .cogxCounterVideo_dasTrackvid
        .cogxCounterVideo_dasTrackvid_path
    """
    
    def __init__(self, loc):
        super().__init__(loc)
        
        properties = {
            "video":                        "video.mp4",                                                                  # raw input video (unaltered)
            "prompt":                       "prompt.txt",                                                                 # original text prompt
            "cogxCounterVideo":             "firstLastInterp_Jack2000.mp4",                                               # counterfactual video generated by CogVideoX with inpainting model
            "video_dasTrackvid":            "video.mp4__DiffusionAsShaderCondition/tracking_video.mp4",                   # Diffusion-as-Shader 3d Point Tracking Video
            "cogxCounterVideo_dasTrackvid": "firstLastInterp_Jack2000.mp4__DiffusionAsShaderCondition/tracking_video.mp4" # Diffusion-as-Shader 3d Point Tracking Video for the counterfactual video
        }
        
        self._create_properties(properties)

class GsDataset:
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]
    
    def __repr__(self):
        return f'{type(self).__name__}(len={len(self)})'


class RawYoutubeDataset(GsDataset):
    def __init__(self):
        self.video_urls, self.prompt_urls = rp.list_transpose(youtube_gs_pairs())

    def __len__(self):
        return len(self.video_urls)

    def __getitem__(self, i):
        return RawYoutubeGsSample(
            rp.get_file_name(self.video_urls[i], include_file_extension=False),
            self.video_urls[i],
            self.prompt_urls[i],
        )

class ProcessedYoutubeDataset(GsDataset):
    def __init__(self):
        self.sample_urls = processed_youtube_gsls()

    def __len__(self):
        return len(self.sample_urls)

    def __getitem__(self, i):
        return ProcessedYoutubeGsSample(
            self.sample_urls[i]
        )