import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

import sys
from icecream import ic
import rp

sys.path.append(rp.get_parent_folder(__file__,levels=2))
from source.video_augmentor import augment_videos


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__()

        if random_flip:
            print("CHEESE NOOO NON ON ONONONON ONONAOI SDOPIUAHSD AUSHD OIUASDHUOIAISDOIHUASDIOAHIOUS DHIOUAAAAAAAAAAAAAAAAAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHHHHHHHHHHH\n"*1000)
        else:
            print("nah fam we good")

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(random_flip)
                # if random_flip
                # else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._preprocess_video(self.video_paths[index])

            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            return {
                "prompt": prompt_embeds,
                "image": image_latents,
                "video": video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            image, video, _ = self._preprocess_video(self.video_paths[index])

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": image,
                "video": video,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
            }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.

        If returning latent/embedding, returns a [F, C, H, W] latent, and the prompt embedding of shape [S, D].
        F, C, H and W are the frames, channels, height and width of the latent, and S, D are the sequence length
        and embedding dimension of prompt embeddings.
        """
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)

            indices = list(range(0, video_num_frames, video_num_frames // self.max_num_frames))
            frames = video_reader.get_batch(indices)
            frames = frames[: self.max_num_frames].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()
            frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, embeds


class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]

class VideoDatasetWithResizingTracking(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.tracking_column         = kwargs.pop("tracking_column"        , None)
        self.counter_tracking_column = kwargs.pop("counter_tracking_column", None)
        self.counter_video_column    = kwargs.pop("counter_video_column"   , None)

        assert self.tracking_column         is not None
        assert self.counter_tracking_column is not None
        assert self.counter_video_column    is not None

        super().__init__(*args, **kwargs)

    def _preprocess_video(
        self,
        path: Path,
        tracking_path: Path,
        counter_tracking_path: Path,
        counter_video_path: Path,
    ) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path, tracking_path)
        else:

            audit_metadata = []

            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            tracking_reader = decord.VideoReader(uri=tracking_path.as_posix())
            tracking_frames = tracking_reader.get_batch(frame_indices)
            tracking_frames = tracking_frames[:nearest_frame_bucket].float()
            tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
            tracking_frames_resized = torch.stack([resize(tracking_frame, nearest_res) for tracking_frame in tracking_frames], dim=0)
            tracking_frames = torch.stack([self.video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)

            counter_tracking_reader = decord.VideoReader(uri=counter_tracking_path.as_posix()) 
            counter_tracking_frames = counter_tracking_reader.get_batch(frame_indices)
            counter_tracking_frames = counter_tracking_frames[:nearest_frame_bucket].float()
            counter_tracking_frames = counter_tracking_frames.permute(0, 3, 1, 2).contiguous()
            counter_tracking_frames_resized = torch.stack([resize(counter_tracking_frame, nearest_res) for counter_tracking_frame in counter_tracking_frames], dim=0)
            counter_tracking_frames = torch.stack([self.video_transforms(counter_tracking_frame) for counter_tracking_frame in counter_tracking_frames_resized], dim=0)

            counter_video_reader = decord.VideoReader(uri=counter_video_path.as_posix())
            counter_video_frames = counter_video_reader.get_batch(frame_indices)
            counter_video_frames = counter_video_frames[:nearest_frame_bucket].float()
            counter_video_frames = counter_video_frames.permute(0, 3, 1, 2).contiguous()
            counter_video_frames_resized = torch.stack([resize(counter_video_frame, nearest_res) for counter_video_frame in counter_video_frames], dim=0)
            counter_video_frames = torch.stack([self.video_transforms(counter_video_frame) for counter_video_frame in counter_video_frames_resized], dim=0)

            #Only 3 variants for each video please! Easier to cache!
            rp.seed_all(rp.millis()%3 + rp.get_sha256_hash(str(path).encode(),format='int')%10000)

            assert len(frames)==len(tracking_frames)==len(counter_video_frames)==len(counter_tracking_frames),f'NOT ALL LENGTHS ARE EQUAL: len(frames)={len(frames)} AND len(tracking_frames)={len(tracking_frames)} AND len(counter_video_frames)={len(counter_video_frames)} AND len(counter_tracking_frames)={len(counter_tracking_frames)}'


            #VIDEO SPEED AUGMENTATION. IT'S CRUDE RIGHT NOW. BUT I NEED TO PROVE IT UNDERSTANDS FIRST/LAST FRAME IS NOT ALL THERE IS...
            #NOTE: WE have a finite number of speeds to best benefit from our cache!
            def regular_speed(video):
                return video
            #
            def half_speed(video):
                return rp.resize_list(video, len(video)*2)[:len(video)]
            #
            def two_thirds_speed(video):
                return rp.resize_list(video, int(len(video)*1.5))[:len(video)]
            #
            def half_speed_second_half(video):
                return rp.resize_list(video, len(video)*2)[len(video):]
            #
            def half_speed_middle(video):
                return rp.resize_list(video, len(video)*2)[len(video)//2:len(video)//2+len(video)]
            #
            def two_thirds_speed_later_half(video):
                return rp.resize_list(video, int(len(video)*1.5))[len(video):]
            #
            def two_thirds_speed_middle(video):
                extended = rp.resize_list(video, int(len(video)*1.5))
                start = int(len(extended)*1/6)
                return extended[start:start+len(video)]            
            #
            RANDOM_SPEED = [
                regular_speed,
                half_speed,
                two_thirds_speed,
                half_speed_second_half,
                half_speed_middle,
                two_thirds_speed_later_half,
                two_thirds_speed_middle,
            ]
            RANDOM_SPEED = rp.random_element(RANDOM_SPEED)
            if rp.random_chance(.5):
                #Sometimes I want even less overlap...
                VIDEO_SPEED = rp.random_element(RANDOM_SPEED)

            #At least one will be full speed
            VIDEO_SPEED, COUNTER_SPEED = regular_speed, RANDOM_SPEED
            if rp.random_chance():
                VIDEO_SPEED, COUNTER_SPEED = [VIDEO_SPEED, COUNTER_SPEED][::-1]

            if rp.random_chance(.2):
                #20% of the time let's make them the same video for perfect correspondences
                counter_video_frames    = 0 + frames
                counter_tracking_frames = 0 + tracking_frames
                audit_metadata+=['Swap']

            rp.fansi_print(f'dataset.py: VIDEO_SPEED={VIDEO_SPEED.__name__}      COUNTER_SPEED={COUNTER_SPEED.__name__}','white blue on black gray italic')
            audit_metadata+=[f'GT_SPEED={VIDEO_SPEED.__name__}',f'CTR_SPEED={COUNTER_SPEED.__name__}']
            #
            assert len(frames)==len(tracking_frames)==len(counter_video_frames)==len(counter_tracking_frames),f'NOT ALL LENGTHS ARE EQUAL: len(frames)={len(frames)} AND len(tracking_frames)={len(tracking_frames)} AND len(counter_video_frames)={len(counter_video_frames)} AND len(counter_tracking_frames)={len(counter_tracking_frames)}'

            original_frames                  = frames
            original_tracking_frames         = tracking_frames
            original_counter_video_frames    = counter_video_frames
            original_counter_tracking_frames = counter_tracking_frames


            frames          = VIDEO_SPEED(frames)
            tracking_frames = VIDEO_SPEED(tracking_frames)
            #
            counter_video_frames    = COUNTER_SPEED(counter_video_frames)
            counter_tracking_frames = COUNTER_SPEED(counter_tracking_frames)

            assert len(frames)==len(tracking_frames)==len(counter_video_frames)==len(counter_tracking_frames),f'NOT ALL LENGTHS ARE EQUAL: len(frames)={len(frames)} AND len(tracking_frames)={len(tracking_frames)} AND len(counter_video_frames)={len(counter_video_frames)} AND len(counter_tracking_frames)={len(counter_tracking_frames)}'


            #Random Reversals
            if rp.random_chance(.2):
                rp.fansi_print(f"dataset.py: REVERSING GROUND TRUTH", "white blue on black gray italic")
                frames          = frames         .flip(0)
                tracking_frames = tracking_frames.flip(0)
                audit_metadata+=['REV-GT']
            if rp.random_chance(.3):
                rp.fansi_print(f"dataset.py: REVERSING COUNTERFACTUALS", "white blue on black gray italic")
                counter_video_frames    = counter_video_frames   .flip(0)
                counter_tracking_frames = counter_tracking_frames.flip(0)
                audit_metadata+=['REV-CTR']

            #Random Sliding Crops
            if rp.random_chance(.2):
                rp.fansi_print(f'RANDOM SLIDING CROP: frames.shape={frames.shape}  tracking_frames={tracking_frames.shape}','green green yellow')
                frames, tracking_frames = augment_videos([frames, tracking_frames])
                audit_metadata+=['AUG-GT']
            if rp.random_chance(.2):
                rp.fansi_print(f'RANDOM SLIDING CROP: frames.shape={frames.shape}  tracking_frames={tracking_frames.shape}','green green yellow')
                counter_video_frames, counter_tracking_frames = augment_videos([counter_video_frames, counter_tracking_frames])
                audit_metadata+=['AUG-CTR']

            # if rp.random_chance(1/200):
            if rp.random_chance(1/200):
                try:
                    dataset_audit_path = f".random_dataset_samples/sample_{rp.random_int(100)}.pkl" #Don't save a total of more than 100 samples so we have no disk leaks
                    dataset_audit_path = rp.get_absolute_path(dataset_audit_path)
                    rp.fansi_print(f"SAVING RANDOM DATASET SAMPLE TO {dataset_audit_path}", "green red white bold underdouble")
                    sample = image, frames, tracking_frames, counter_tracking_frames, counter_video_frames
                    rp.object_to_file(sample, dataset_audit_path)

                    audit_videos=[rp.as_numpy_video(x/2+.5) for x in sample]
                    audit_videos=rp.labeled_videos(audit_videos,'Image Frames Tracks CounterTracks CounterFrames'.split())
                    audit_videos=rp.horizontally_concatenated_videos(audit_videos)
                    orig_videos = rp.horizontally_concatenated_videos(
                        [rp.as_numpy_video(x/2+.5) for x in [original_frames*0, original_frames, original_tracking_frames, original_counter_tracking_frames, original_counter_video_frames]]
                    )
                    audit_videos=rp.labeled_images(audit_videos,"  :  ".join(audit_metadata))
                    
                    print(rp.save_video_mp4(rp.vertically_concatenated_videos(audit_videos,orig_videos),dataset_audit_path+'.mp4',framerate=20))

                except Exception:
                    rp.print_stack_trace()
            
            return image, frames, tracking_frames, counter_tracking_frames, counter_video_frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]
    
    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)
        tracking_path = self.data_root.joinpath(self.tracking_column)
        counter_tracking_path = self.data_root.joinpath(self.counter_tracking_column)
        counter_video_path = self.data_root.joinpath(self.counter_video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                f"Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts. prompt_path={prompt_path}"
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                f"Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory. video_path={video_path}"
            )
        if not tracking_path.exists() or not tracking_path.is_file():
            raise ValueError(
                f"Expected `--tracking_column` to be path to a file in `--data_root` containing line-separated tracking information. tracking_path={tracking_path}"
            )
        if not counter_tracking_path.exists() or not counter_tracking_path.is_file():
            raise ValueError(
                f"Expected `--counter_tracking_column` to be path to a file in `--data_root` containing line-separated counter_tracking information. counter_tracking_path={counter_tracking_path}"
            )
        if not counter_video_path.exists() or not counter_video_path.is_file():
            raise ValueError(
                f"Expected `--counter_video_column` to be path to a file in `--data_root` containing line-separated counter_video information. counter_video_path={counter_video_path}"
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
            
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(tracking_path, "r", encoding="utf-8") as file:
            tracking_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(counter_tracking_path, "r", encoding="utf-8") as file:
            counter_tracking_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(counter_video_path, "r", encoding="utf-8") as file:
            counter_video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        self.tracking_paths = tracking_paths
        self.counter_tracking_paths = counter_tracking_paths
        self.counter_video_paths = counter_video_paths
        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        tracking_paths = df[self.tracking_column].tolist()
        counter_tracking_paths = df[self.counter_tracking_column].tolist()
        counter_video_paths = df[self.counter_video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]
        tracking_paths = [self.data_root.joinpath(line.strip()) for line in tracking_paths]
        counter_tracking_paths = [self.data_root.joinpath(line.strip()) for line in counter_tracking_paths]
        counter_video_paths = [self.data_root.joinpath(line.strip()) for line in counter_video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found at least one path that is not a valid file."
            )

        self.tracking_paths = tracking_paths
        self.counter_tracking_paths = counter_tracking_paths
        self.counter_video_paths = counter_video_paths
        return prompts, video_paths
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        while True:
            try:
                if self.load_tensors:
                    assert False, 'Currently we do not support load_tensors'
                    # image_latents, video_latents, prompt_embeds = self._preprocess_video(self.video_paths[index], self.tracking_paths[index])
                    #
                    # # The VAE's temporal compression ratio is 4.
                    # # The VAE's spatial compression ratio is 8.
                    # latent_num_frames = video_latents.size(1)
                    # if latent_num_frames % 2 == 0:
                    #     num_frames = latent_num_frames * 4
                    # else:
                    #     num_frames = (latent_num_frames - 1) * 4 + 1
                    #
                    # height = video_latents.size(2) * 8
                    # width = video_latents.size(3) * 8
                    #
                    # return {
                    #     "prompt": prompt_embeds,
                    #     "image": image_latents,
                    #     "video": video_latents,
                    #     "tracking_map": tracking_map,
                    #     "video_metadata": {
                    #         "num_frames": num_frames,
                    #         "height": height,
                    #         "width": width,
                    #     },
                    # }
                else:
                    image, video, tracking_map, counter_tracking_map, counter_video_map, _ = self._preprocess_video(
                        self.video_paths[index],
                        self.tracking_paths[index],
                        self.counter_tracking_paths[index],
                        self.counter_video_paths[index],
                    )
                    rp.fansi_print(f'Good sample at index={index}','green green italic bold')
                    return {
                        "prompt": self.id_token + self.prompts[index],
                        "image": image,
                        "video": video,
                        "tracking_map": tracking_map,
                        "counter_tracking_map": counter_tracking_map,
                        "counter_video_map": counter_video_map,
                        "video_metadata": {
                            "num_frames": video.shape[0],
                            "height": video.shape[2],
                            "width": video.shape[3],
                        },
                    }
            except Exception as e:
                rp.fansi_print(f'DATASET ERROR! index={index}    video_path={self.video_paths[index]}    tracking_path={self.tracking_paths[index]}    {e}','red red bold on black black undercurl')
                # rp.print_stack_trace()
                index=rp.random_index(self)
            
    
    def _load_preprocessed_latents_and_embeds(self, path: Path, tracking_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        rp.fansi_print(f"WHY THE FUCK ARE WE HERE??? path={path} tracking_path={tracking_path}",'bold green green on red red')
        assert False, "WHY TTGE FUCK"

        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"
        
        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        tracking_map_path = path.parent.parent.joinpath("tracking_map")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")
        
        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or not tracking_map_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains folders named `video_latents`, `prompt_embeds`, and `tracking_map`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )
        
        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        tracking_map_filepath = tracking_map_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)
        
        if not video_latent_filepath.is_file() or not embeds_filepath.is_file() or not tracking_map_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            tracking_map_filepath = tracking_map_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} or {tracking_map_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )
        
        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        tracking_map = torch.load(tracking_map_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)
        
        return images, latents, tracking_map, embeds

class VideoDatasetWithResizeAndRectangleCrop(VideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self, data_source: VideoDataset, batch_size: int = 8, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolutions}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []
