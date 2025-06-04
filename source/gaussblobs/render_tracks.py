import rp
import torch
import numpy as np
from einops import rearrange
import numba

def subdivide_track_grids(tracks, new_TH, new_TW):
    #Takes a T TH TW XYZV grid, does subdivision along the X and Y axes.
    #Output shape is T new_TH new_TW XYZV

    track_grids = rearrange(tracks, 'T (TH TW) XYZV -> T TH TW XYZV', TH=TH, TW=TW)

    upsampled = rearrange(track_grids, 'T TH TW XYZV -> T XYZV TH TW')
    upsampled = torch.nn.functional.interpolate(
        upsampled,
        size=(new_TH, new_TW),
        mode='bilinear',
        align_corners=True
    )
    upsampled = rearrange(upsampled, 'T XYZV TH TW -> T TH TW XYZV')

    return upsampled

def soak_track_grids(track_grids, video):
    #Concat color channels to the XYZV from the video at appropriate places

    T, TH, TW, C, XYZV = rp.validate_tensor_shapes(
        track_grids        = 'torch: T TH TW XYZV',
        video              = 'torch: T C VH VW',
        C = video.shape[1],
        XYZV = 4,
        return_dims = 'T TH TW C XYZV',
    )

    soaked_track_grids = []
    for track_grid, frame in zip(track_grids, video):
        assert track_grid.shape == (TH, TW, XYZV)

        soaked_image = rp.torch_remap_image(
            image=frame,
            x=track_grid[:, :, 0],
            y=track_grid[:, :, 1],
            use_cached_meshgrid=True,
            interp="bilinear",
        )
        assert soaked_image.shape == (C, TH, TW)

        soaked_track_grid = torch.cat(
            [
                track_grid,
                rearrange(soaked_image, "C TH TW -> TH TW C"),
            ],
            dim=2,
        )
        assert soaked_track_grid.shape == (TH, TW, XYZV + C)
        soaked_track_grids.append(soaked_track_grid)

    soaked_track_grids = torch.stack(soaked_track_grids)

    assert soaked_track_grids.shape == (T, TH, TW, XYZV + C)

    return soaked_track_grids

@numba.njit(parallel=True)
def _draw_soaked_track_grids_numba(soaked_track_grids_np, VH, VW):
    # Numba implementation for better performance
    T, TH, TW, XYZVC = soaked_track_grids_np.shape
    XYZV = 4
    C = XYZVC - XYZV
    Xi, Yi, Zi, Vi, Ci = 0, 1, 2, 3, 4

    video_np   = np.zeros((T, C, VH, VW),               dtype=np.float32)
    video_np_z = np.full ((T,    VH, VW), float('inf'), dtype=np.float32)

    for t in numba.prange(T):
        for ty in range(TH):
            for tx in range(TW):
                vx = int(soaked_track_grids_np[t, ty, tx, Xi])
                vy = int(soaked_track_grids_np[t, ty, tx, Yi])
                vz =     soaked_track_grids_np[t, ty, tx, Zi]
                vv = int(soaked_track_grids_np[t, ty, tx, Vi])

                old_z = video_np_z[t, vy, vx]

                if vz<old_z and vv and 0 <= vy < VH and 0 <= vx < VW:
                    for c in range(C):
                        video_np[t, c, vy, vx] = soaked_track_grids_np[t, ty, tx, Ci + c]
                    video_np_z[t, vy, vx] = vz

    return video_np

@numba.njit(parallel=True)
def _draw_soaked_track_grid_gaussians_numba(soaked_track_grids_np, VH, VW, sigma):
    # Numba implementation for gaussian blob rendering with alpha compositing
    T, TH, TW, XYZVC = soaked_track_grids_np.shape
    XYZV = 4
    C = XYZVC - XYZV
    Xi, Yi, Zi, Vi, Ci = 0, 1, 2, 3, 4

    video_np = np.zeros((T, C, VH, VW), dtype=np.float32)

    # Precompute gaussian radius (3 sigma cutoff for efficiency)
    radius = int(np.ceil(3.0 * sigma))

    for t in numba.prange(T):
        for ty in range(TH):
            for tx in range(TW):
                center_x = soaked_track_grids_np[t, ty, tx, Xi]
                center_y = soaked_track_grids_np[t, ty, tx, Yi]
                vz = soaked_track_grids_np[t, ty, tx, Zi]
                vv = int(soaked_track_grids_np[t, ty, tx, Vi])

                if not vv:
                    continue

                # Draw gaussian blob
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        px = int(center_x + dx)
                        py = int(center_y + dy)

                        if 0 <= py < VH and 0 <= px < VW:
                            # Calculate gaussian weight
                            dist_sq = dx * dx + dy * dy
                            gauss_weight = np.exp(-dist_sq / (2.0 * sigma * sigma))

                            # Alpha composite all channels
                            for c in range(C):
                                color_value = soaked_track_grids_np[t, ty, tx, Ci + c]
                                alpha = color_value * gauss_weight if C != 4 else soaked_track_grids_np[t, ty, tx, Ci + 3] * gauss_weight

                                # Alpha compositing: new_color = old_color * (1 - alpha) + new_color * alpha
                                one_minus_alpha = 1.0 - alpha
                                video_np[t, c, py, px] = video_np[t, c, py, px] * one_minus_alpha + color_value * alpha

    return video_np

def draw_soaked_track_grids(soaked_track_grids, VH:int, VW:int):
    # Convert to numpy for numba processing, then back to torch
    # Force float32 to avoid float16 issues on macOS
    soaked_track_grids_np = soaked_track_grids.cpu().float().numpy()
    video_np = _draw_soaked_track_grids_numba(soaked_track_grids_np, VH, VW)
    video = torch.from_numpy(video_np)

    C = soaked_track_grids.shape[3] - 4  # XYZV is 4, so C = total - 4
    rp.validate_tensor_shapes(
        soaked_track_grids    = "torch: T TH TW XYZVC   ",
        soaked_track_grids_np = "numpy: T TH TW XYZVC   ",
        video_np              = "numpy: T C VH VW       ",
        video                 = "torch: T C VH VW       ",
        XYZV=4, C=C,
    )

    return video

def draw_soaked_track_grid_gaussians(soaked_track_grids, VH:int, VW:int, sigma:float):
    # Convert to numpy for numba processing, then back to torch
    # Force float32 to avoid float16 issues on macOS
    soaked_track_grids_np = soaked_track_grids.cpu().float().numpy()
    video_np = _draw_soaked_track_grid_gaussians_numba(soaked_track_grids_np, VH, VW, sigma)
    video = torch.from_numpy(video_np)

    C = soaked_track_grids.shape[3] - 4  # XYZV is 4, so C = total - 4
    rp.validate_tensor_shapes(
        soaked_track_grids    = "torch: T TH TW XYZVC   ",
        soaked_track_grids_np = "numpy: T TH TW XYZVC   ",
        video_np              = "numpy: T C VH VW       ",
        video                 = "torch: T C VH VW       ",
        XYZV=4, C=C,
    )

    return video

def random_7_gaussians_video(tracks, counter_tracks, VH, VW, sigma=5.0, seed=42, blob_colors=None):
    """
    Select random tracks and render them as gaussian blobs with distinct colors.

    Args:
        tracks: torch tensor [T, N, XYZV] - original video tracks
        counter_tracks: torch tensor [T, N, XYZV] - counterfactual video tracks
        VH, VW: output video height and width
        sigma: gaussian blob size
        seed: random seed for reproducible track selection
        blob_colors: list of color tensors, defaults to 7 RGBA colors

    Returns:
        tuple of (video_gaussians, counter_video_gaussians) - both [T, C, VH, VW] where C is determined by color channels
    """
    T, N, XYZV = tracks.shape

    # Set default colors if none provided
    if blob_colors is None:
        blob_colors = [
            [1, 0, 0, 1],  # red
            [0, 1, 0, 1],  # green
            [0, 0, 1, 1],  # blue
            [0, 1, 1, 1],  # cyan
            [1, 0, 1, 1],  # magenta
            [1, 1, 0, 1],  # yellow
            [1, 1, 1, 1],  # white
        ]

    # Convert to tensor and get dimensions
    colors = torch.tensor(blob_colors, dtype=tracks.dtype)
    num_blobs, C = colors.shape

    # Set random seed for reproducible selection
    torch.manual_seed(seed)

    # Select random track indices based on number of colors
    selected_indices = torch.randperm(N)[:num_blobs]

    # Create black background videos with appropriate number of channels
    video_gaussians         = torch.zeros((T, C, VH, VW), dtype=tracks.dtype)
    counter_video_gaussians = torch.zeros((T, C, VH, VW), dtype=tracks.dtype)

    # Process each selected track
    for i, track_idx in enumerate(selected_indices):
        color = colors[i]

        # Extract single track for both videos
        single_track = tracks[:, track_idx:track_idx+1, :]  # [T, 1, XYZV]
        single_counter_track = counter_tracks[:, track_idx:track_idx+1, :]  # [T, 1, XYZV]

        # Reshape to grid format (1x1 grid since it's a single track)
        track_grid = single_track.view(T, 1, 1, XYZV)
        counter_track_grid = single_counter_track.view(T, 1, 1, XYZV)

        # Create soaked track grids with the assigned color
        soaked_track_grid = torch.cat([
            track_grid,  # XYZV
            color.view(1, 1, 1, C).expand(T, 1, 1, C)  # Color channels
        ], dim=3)  # [T, 1, 1, XYZV+C]

        soaked_counter_track_grid = torch.cat([
            counter_track_grid,  # XYZV
            color.view(1, 1, 1, C).expand(T, 1, 1, C)  # Color channels
        ], dim=3)  # [T, 1, 1, XYZV+C]

        # Draw gaussians for this track
        track_video = draw_soaked_track_grid_gaussians(soaked_track_grid, VH, VW, sigma)
        counter_track_video = draw_soaked_track_grid_gaussians(soaked_counter_track_grid, VH, VW, sigma)

        # Alpha composite onto the accumulating videos using the last channel as alpha if available
        if C == 4:  # RGBA case - use alpha channel
            track_alpha = track_video[:, 3:4, :, :]
            counter_alpha = counter_track_video[:, 3:4, :, :]
        else:  # For other channels, use binary mask
            track_alpha = (track_video > 0).any(dim=1, keepdim=True).float()
            counter_alpha = (counter_track_video > 0).any(dim=1, keepdim=True).float()

        # Alpha compositing: result = old * (1 - alpha) + new * alpha
        video_gaussians = video_gaussians * (1 - track_alpha) + track_video * track_alpha
        counter_video_gaussians = counter_video_gaussians * (1 - counter_alpha) + counter_track_video * counter_alpha

    rp.validate_tensor_shapes(
        tracks                      = "torch: T N XYZV           ",
        counter_tracks              = "torch: T N XYZV           ",
        colors                      = "torch: num_blobs C        ",
        video_gaussians             = "torch: T C VH VW          ",
        counter_video_gaussians     = "torch: T C VH VW          ",
        XYZV=4, C=C, num_blobs=num_blobs,
    )

    return video_gaussians, counter_video_gaussians

def video_warp(video, counter_video, video_tracks, counter_tracks):
    """
    Perform video warping and create all the standard visualization outputs.

    Args:
        video: torch tensor [T, C, VH, VW] - original video
        counter_video: torch tensor [T, C, VH, VW] - counterfactual video
        video_tracks: torch tensor [T, N, XYZV] - original video tracks
        counter_tracks: torch tensor [T, N, XYZV] - counterfactual video tracks

    Returns:
        easydict containing all output videos and intermediate results
    """
    T, C, VH, VW = video.shape
    T, N, XYZV = video_tracks.shape

    # Upscale the tracks
    THS = VH//16  # Tracks height subdivided
    TWS = VW//16  # Tracks width subdivided

    video_track_grids = subdivide_track_grids(video_tracks, THS, TWS)
    counter_track_grids = subdivide_track_grids(counter_tracks, THS, TWS)

    # Create soaked track grids (tracks with colors from video)
    soaked_track_grids = soak_track_grids(video_track_grids, video)
    # drawn_video = draw_soaked_track_grids(soaked_track_grids, VH, VW)
    drawn_video = draw_soaked_track_grid_gaussians(soaked_track_grids, VH, VW, sigma=5)

    # Create counter version - inherit colors but use counter positions
    counter_soaked_track_grids = soaked_track_grids + 0  # Inherit the colors
    counter_soaked_track_grids[:,:,:,:2] = counter_track_grids[:,:,:,:2]  # Inherit the new XY
    counter_soaked_track_grids[:,:,:,3] *= counter_track_grids[:,:,:,3]  # Intersection of visibility

    # counter_soaked_track_grids[:,:,:,3] = 1  # No invisibility anywhere

    # Draw counter video
    # counter_drawn_video = draw_soaked_track_grids(counter_soaked_track_grids, VH, VW)
    counter_drawn_video = draw_soaked_track_grid_gaussians(counter_soaked_track_grids, VH, VW, sigma=5)

    # Convert for display
    counter_drawn_video_np = rp.as_numpy_images(counter_drawn_video)
    # counter_drawn_video_np = rp.with_alpha_checkerboards(counter_drawn_video_np)
    counter_drawn_video_np = rp.as_rgb_images(counter_drawn_video_np)

    # Create overlay
    counter_drawn_video_alpha = counter_drawn_video[:,3:4,:,:]
    counter_drawn_video_overlaid = (
        1 - counter_drawn_video_alpha
    ) * counter_video + counter_drawn_video_alpha * counter_drawn_video

    # Create preview video
    preview_video = rp.vertically_concatenated_videos(
        rp.as_numpy_videos([
            video,
            counter_drawn_video_np,
            counter_drawn_video_overlaid,
            counter_video,
        ])
    )

    T, C, VH, VW = video.shape
    rp.validate_tensor_shapes(
        video                         = "torch: T C VH VW      ",
        counter_video                 = "torch: T C VH VW      ",
        video_tracks                  = "torch: T N XYZV       ",
        counter_tracks                = "torch: T N XYZV       ",
        video_track_grids             = "torch: T THS TWS XYZV ",
        counter_track_grids           = "torch: T THS TWS XYZV ",
        soaked_track_grids            = "torch: T THS TWS XYZVC",
        counter_soaked_track_grids    = "torch: T THS TWS XYZVC",
        drawn_video                   = "torch: T C VH VW      ",
        counter_drawn_video           = "torch: T C VH VW      ",
        counter_drawn_video_overlaid  = "torch: T C VH VW      ",
        C=C, XYZV=4, XYZVC=4+C,
    )

    return rp.gather_vars(
        "video_track_grids",
        "counter_track_grids",
        "soaked_track_grids",
        "counter_soaked_track_grids",
        "drawn_video",
        "counter_drawn_video",
        "counter_drawn_video_np",
        "counter_drawn_video_overlaid",
        "preview_video",
    )

def draw_blobs_videos(video, counter_video, video_tracks, counter_tracks, blob_colors=None):
    """
    Draw colored gaussian blob videos for selected tracks.

    Args:
        video: torch tensor [T, C, VH, VW] - original video
        counter_video: torch tensor [T, C, VH, VW] - counterfactual video
        video_tracks: torch tensor [T, N, XYZV] - original video tracks
        counter_tracks: torch tensor [T, N, XYZV] - counterfactual video tracks
        blob_colors: list of colors, defaults to 7 standard colors

    Returns:
        easydict containing gaussian videos and overlay visualizations
    """
    # Generate gaussian blob videos first to determine channel count
    video_gaussians, counter_video_gaussians = random_7_gaussians_video(
        video_tracks, counter_tracks, video.shape[2], video.shape[3], sigma=16, blob_colors=blob_colors
    )

    T, C, VH, VW = video_gaussians.shape

    # Resize input videos to match gaussian channel count
    video = rp.crop_tensor(video, (T, C, VH, VW))
    counter_video = rp.crop_tensor(counter_video, (T, C, VH, VW))

    # Convert to numpy for display
    video_gaussians_np         = rp.as_numpy_images(video_gaussians)
    counter_video_gaussians_np = rp.as_numpy_images(counter_video_gaussians)

    # Create overlays: replace video pixels with gaussian pixels wherever gaussians are not black
    video_mask   = (video_gaussians         > 0).any(dim=1, keepdim=True).expand(-1, C, -1, -1)
    counter_mask = (counter_video_gaussians > 0).any(dim=1, keepdim=True).expand(-1, C, -1, -1)

    video_on_gaussians   = torch.where(video_mask  , video_gaussians        , video        )
    counter_on_gaussians = torch.where(counter_mask, counter_video_gaussians, counter_video)

    # Convert overlays to numpy
    video_on_gaussians_np   = rp.as_numpy_images(video_on_gaussians  )
    counter_on_gaussians_np = rp.as_numpy_images(counter_on_gaussians)

    # Create side-by-side visualization
    gaussian_preview = rp.horizontally_concatenated_videos(
        counter_on_gaussians_np,
        video_on_gaussians_np,
    )

    rp.validate_tensor_shapes(
        video_tracks                 = "torch: T N XYZV ",
        counter_tracks               = "torch: T N XYZV ",
        video_gaussians              = "torch: T C VH VW",
        counter_video_gaussians      = "torch: T C VH VW",
        video_mask                   = "torch: T C VH VW",
        counter_mask                 = "torch: T C VH VW",
        video                        = "torch: T C VH VW",
        counter_video                = "torch: T C VH VW",
        video_on_gaussians           = "torch: T C VH VW",
        counter_on_gaussians         = "torch: T C VH VW",
        video_gaussians_np           = "numpy: T VH VW C",
        counter_video_gaussians_np   = "numpy: T VH VW C",
        video_on_gaussians_np        = "numpy: T VH VW C",
        counter_on_gaussians_np      = "numpy: T VH VW C",
        XYZV=4, C=C, VH=VH, VW=VW, T=T,
    )

    return rp.gather_vars(
        "video_gaussians",
        "counter_video_gaussians",
        "video_gaussians_np",
        "counter_video_gaussians_np",
        "video_on_gaussians",
        "counter_on_gaussians",
        "video_on_gaussians_np",
        "counter_on_gaussians_np",
        "gaussian_preview",
    )


#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-7Cxuw5aZAY_405555963_425701967" #GOOD
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-D3HZiCsa_Q_781740353_789776475" #GOOD
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-6v_98wULaw_696440406_708166584" #GOOD
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-mYvWIeIEHE_268812917_274856884" #MEH
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-IexzyFb-rs_313386441_330096967" #MEH
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-6CEIXlDqN8_13713228_22952664" #MEH
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-IydcPzQDtc_148703392_166799248" #MEH
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-6583XLWavs_576454762_602615422"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-GykhXGPdCY_1043306_27011439"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-1pwRBTrh3w_709851076_714940344"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-IMTp3vElAw_39488821_46186036"#highway
sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-ALNgmWCI9o_376096922_383032458"#Steak Man
# sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-MZovLVMlp8_542531041_560826468"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-HmktmTdFg8_296236407_301336408"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-Ss4S_5u1Kc_366437715_386234521"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-4xvcgJ31Y8_488281696_497364031"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-Rh8clIdruw_757064895_794499732"
# sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-23IOYAVDSg_368956452_379960988"
# sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-4-9WV4XlKc_326403629_333216012"
# sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-04eUuArz9Q_264584202_272741097"
# sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-1gP0sq1OOM_205893836_212747865"
# sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-2nkJuX1jX0_70008500_93177610"
# sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-3U-VPjuKsU_151380105_157242291"
# sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-6WxHTD0MNs_14963716_22880696"
# sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-6Zho9MUmIY_415205378_420241632"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-9n50IYeE0w_197856754_211712029"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-F3Y_ea2tFs_19320398_30174613"#GOOD WITH NO OCCLUSION
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-FD1M0oUqo8_611431840_619738817"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-GASw02SYA4_136939096_145918349"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-HSDibXhB-A_122322552_128332443"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-HttH6P3-Ko_140863713_145879539"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-KtDSfBxDWI_28771129_34863225"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-OWFTTv7An8_26501966_37237520"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-Oz-fKczGkE_75389592_83341653"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-PRF_KbhO2c_7470714_14342441"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-WTjFTxpWe8_114299982_129491545"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-ZZpM1eWAf4_445739054_463467123"
#sample_dir = "/Users/burgert/CleanCode/Sandbox/youtube-ds/-_GuKxi7u8A_107214497_114095456"

with rp.SetCurrentDirectoryTemporarily(sample_dir):

    video_tracks  = rp.as_easydict(torch.load("video.mp4__DiffusionAsShaderCondition/video_tracks_spatracker.pt"                                      , map_location="cpu"))
    counter_tracks= rp.as_easydict(torch.load("firstLastInterp_Jack2000.mp4__DiffusionAsShaderCondition/firstLastInterp_Jack2000_tracks_spatracker.pt", map_location="cpu"))

    video_tracks   = torch.concat([video_tracks  .tracks, video_tracks  .visibility[:,:,None]],2)
    counter_tracks = torch.concat([counter_tracks.tracks, counter_tracks.visibility[:,:,None]],2)

    video         = rp.load_video("video.mp4"            , use_cache=True)
    counter_video = rp.load_video("firstLastInterp_Jack2000.mp4", use_cache=True)

    video = rp.resize_images(video,size=(480,720))
    video=rp.resize_list(video,49)

    video         = rp.as_torch_images(video        )
    counter_video = rp.as_torch_images(counter_video)

    # Add alpha channel to videos (set to 1.0)
    video = torch.cat([video, torch.ones_like(video[:, :1])], dim=1)
    counter_video = torch.cat([counter_video, torch.ones_like(counter_video[:, :1])], dim=1)

    #counter_video = counter_video.flip(0)
    #counter_tracks = counter_tracks.flip(0)

    # counter_video = video.flip(0)
    # counter_tracks = video_tracks.flip(0)



#After counting the dots, I found the default spatialtracker results in a 70x70 grid.
TH = 70 #Tracks height
TW = 70 #Tracks width

T, N, VH, VW = rp.validate_tensor_shapes(
    video_tracks   = "torch: T N XYZV",
    counter_tracks = "torch: T N XYZV",
    video          = "torch: T C VH VW",
    counter_video  = "torch: T C VH VW",
    N   = TH * TW,
    XYZV = 4,
    C = video.shape[1],
    return_dims = 'T N VH VW',
    verbose     = 'white white altbw green',
)



# Run video warp visualization
warp_results = video_warp(video, counter_video, video_tracks, counter_tracks)
rp.save_video_mp4(warp_results.preview_video, framerate=20)
rp.display_image_slideshow(warp_results.preview_video)

# Run blob visualization
blob_results = draw_blobs_videos(video, counter_video, video_tracks, counter_tracks)
rp.save_video_mp4(blob_results.gaussian_preview, "gaussian_tracks_visualization.mp4", framerate=20)
rp.display_image_slideshow(blob_results.gaussian_preview)
