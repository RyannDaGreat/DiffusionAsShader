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
    #Concat RGBA to the XYZV from the video at appropriate places

    T, TH, TW, RGBA, XYZV = rp.validate_tensor_shapes(
        track_grids        = 'torch: T TH TW XYZV',
        video              = 'torch: T RGBA VH VW',
        RGBA = 4,
        XYZV = 4,
        return_dims = 'T TH TW RGBA XYZV',
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
        assert soaked_image.shape == (RGBA, TH, TW)

        soaked_track_grid = torch.cat(
            [
                track_grid,
                rearrange(soaked_image, "RGBA TH TW -> TH TW RGBA"),
            ],
            dim=2,
        )
        assert soaked_track_grid.shape == (TH, TW, XYZV + RGBA)
        soaked_track_grids.append(soaked_track_grid)

    soaked_track_grids = torch.stack(soaked_track_grids)

    assert soaked_track_grids.shape == (T, TH, TW, XYZV + RGBA)
    
    return soaked_track_grids
    
@numba.njit(parallel=True)
def _draw_soaked_track_grids_numba(soaked_track_grids_np, VH, VW):
    # Numba implementation for better performance
    T, TH, TW, XYZVRGBA = soaked_track_grids_np.shape
    RGBA = 4
    Xi, Yi, Zi, Vi, RGBAi = 0, 1, 2, 3, 4
    
    video_np   = np.zeros((T, RGBA, VH, VW),               dtype=np.float32)
    video_np_z = np.full ((T,       VH, VW), float('inf'), dtype=np.float32)
    
    for t in numba.prange(T):
        for ty in range(TH):
            for tx in range(TW):
                vx = int(soaked_track_grids_np[t, ty, tx, Xi])
                vy = int(soaked_track_grids_np[t, ty, tx, Yi])
                vz =     soaked_track_grids_np[t, ty, tx, Zi]
                vv = int(soaked_track_grids_np[t, ty, tx, Vi])
                
                old_z = video_np_z[t, vy, vx]
                
                if vz<old_z and vv and 0 <= vy < VH and 0 <= vx < VW:
                    for c in range(RGBA):
                        video_np[t, c, vy, vx] = soaked_track_grids_np[t, ty, tx, RGBAi + c]
                    video_np_z[t, vy, vx] = vz
    
    return video_np

def draw_soaked_track_grids(soaked_track_grids, VH:int, VW:int):
    # Convert to numpy for numba processing, then back to torch
    soaked_track_grids_np = soaked_track_grids.cpu().numpy()
    video_np = _draw_soaked_track_grids_numba(soaked_track_grids_np, VH, VW)
    video = torch.from_numpy(video_np)

    rp.validate_tensor_shapes(
        soaked_track_grids    = "torch: T TH TW XYZVRGBA ",
        soaked_track_grids_np = "numpy: T TH TW XYZVRGBA ",
        video_np              = "numpy: T RGBA VH VW     ",
        video                 = "torch: T RGBA VH VW     ",
        RGBA=4,
    )
    
    return video


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
        
    counter_video = video.flip(0)
    counter_tracks = video_tracks.flip(0)



#After counting the dots, I found the default spatialtracker results in a 70x70 grid.
TH = 70 #Tracks height
TW = 70 #Tracks width

T, N, VH, VW = rp.validate_tensor_shapes(
    video_tracks   = "torch: T N XYZV",
    counter_tracks = "torch: T N XYZV",
    video          = "torch: T RGBA VH VW",
    counter_video  = "torch: T RGBA VH VW",
    N   = TH * TW,
    XYZV = 4,
    RGBA = 4,
    return_dims = 'T N VH VW',
    verbose     = 'white white altbw green',
)



#Upscale the tracks...
THS = VH #1 #Tracks height subdivided
TWS = VW #1 #Tracks width subdivided
print('cheese')
video_track_grids   = subdivide_track_grids(video_tracks  , THS, TWS)
print('choose')
counter_track_grids = subdivide_track_grids(counter_tracks, THS, TWS)
print('choaws')

rp.validate_tensor_shapes(
    counter_track_grids = "torch: T THS TWS XYZV",
    video_track_grids   = "torch: T THS TWS XYZV",
    XYZV=4,
    **rp.gather_vars("T TWS TWS"),
)




soaked_track_grids = soak_track_grids(video_track_grids, video)
drawn_video = draw_soaked_track_grids(soaked_track_grids, VH, VW)

counter_soaked_track_grids = soaked_track_grids + 0 #Inherit the colors
soaked_track_grids[:,:,:,:2]=counter_track_grids[:,:,:,:2] #Inherit the new XY
soaked_track_grids[:,:,:,3]*=counter_track_grids[:,:,:,3] #Intersection of visibility


#soaked_track_grids[:,:,:,3]=1 #No invisiblity anywhere 

#Draw a video from it
counter_drawn_video = draw_soaked_track_grids(soaked_track_grids, VH, VW)

counter_drawn_video_np = rp.as_numpy_images(counter_drawn_video)
counter_drawn_video_np =rp.with_alpha_checkerboards(counter_drawn_video_np)

###

counter_drawn_video_alpha = counter_drawn_video[:,3:4,:,:]
counter_drawn_video_overlaid = (
    1 - counter_drawn_video_alpha
) * counter_video + counter_drawn_video_alpha * counter_drawn_video

preview_video = rp.vertically_concatenated_videos(
    rp.as_numpy_videos(
        [
            video,
            counter_drawn_video_np,
            counter_drawn_video_overlaid,
            counter_video,
        ],
    )
)

rp.save_video_mp4(preview_video,framerate=20)
rp.display_image_slideshow(preview_video)
