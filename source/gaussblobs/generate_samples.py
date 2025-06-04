import torch
import render_tracks
import rp

sample_paths = [
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-7Cxuw5aZAY_405555963_425701967",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-D3HZiCsa_Q_781740353_789776475",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-6v_98wULaw_696440406_708166584",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-mYvWIeIEHE_268812917_274856884",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-IexzyFb-rs_313386441_330096967",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-6CEIXlDqN8_13713228_22952664",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-IydcPzQDtc_148703392_166799248",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-6583XLWavs_576454762_602615422",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-GykhXGPdCY_1043306_27011439",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-1pwRBTrh3w_709851076_714940344",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-IMTp3vElAw_39488821_46186036",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-ALNgmWCI9o_376096922_383032458",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-MZovLVMlp8_542531041_560826468",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-HmktmTdFg8_296236407_301336408",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-Ss4S_5u1Kc_366437715_386234521",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-4xvcgJ31Y8_488281696_497364031",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-Rh8clIdruw_757064895_794499732",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-23IOYAVDSg_368956452_379960988",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-4-9WV4XlKc_326403629_333216012",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-04eUuArz9Q_264584202_272741097",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-1gP0sq1OOM_205893836_212747865",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-2nkJuX1jX0_70008500_93177610",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-3U-VPjuKsU_151380105_157242291",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-6WxHTD0MNs_14963716_22880696",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-6Zho9MUmIY_415205378_420241632",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-9n50IYeE0w_197856754_211712029",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-F3Y_ea2tFs_19320398_30174613",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-FD1M0oUqo8_611431840_619738817",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-GASw02SYA4_136939096_145918349",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-HSDibXhB-A_122322552_128332443",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-HttH6P3-Ko_140863713_145879539",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-KtDSfBxDWI_28771129_34863225",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-OWFTTv7An8_26501966_37237520",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-Oz-fKczGkE_75389592_83341653",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-PRF_KbhO2c_7470714_14342441",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-WTjFTxpWe8_114299982_129491545",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-ZZpM1eWAf4_445739054_463467123",
    "/Users/burgert/CleanCode/Sandbox/youtube-ds/-_GuKxi7u8A_107214497_114095456",
]


@rp.globalize_locals
def main():
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
        video         = torch.cat([video,         torch.ones_like(video[:,         :1])], dim=1)
        counter_video = torch.cat([counter_video, torch.ones_like(counter_video[:, :1])], dim=1)

        counter_video  = counter_video .flip(0)
        counter_tracks = counter_tracks.flip(0)

        # counter_video  = video       .flip(0)
        # counter_tracks = video_tracks.flip(0)

    #After counting the dots, I found the default spatialtracker results in a 70x70 grid.
    TH = 70 #Tracks height
    TW = 70 #Tracks width

    T, N, VH, VW = rp.validate_tensor_shapes(
        video_tracks   = "torch: T N XYZV",
        counter_tracks = "torch: T N XYZV",
        video          = "torch: T C VH VW",
        counter_video  = "torch: T C VH VW",
        N = TH * TW,
        XYZV = 4,
        return_dims = 'T N VH VW',
        verbose     = 'white white altbw green',
    )

    # Run video warp visualization
    #warp_results = video_warp(video, counter_video, video_tracks, counter_tracks)
    #rp.save_video_mp4(warp_results.preview_video, framerate=20)
    #rp.display_image_slideshow(warp_results.preview_video)

    # Run blob visualization
    blob_results = render_tracks.draw_blobs_videos(video, counter_video, video_tracks, counter_tracks, visualize=True,sigma=8)
    rp.save_video_mp4(blob_results.gaussian_preview, get_unique_copy_path("gaussian_tracks_visualization.mp4"), framerate=20)
    #rp.display_image_slideshow(blob_results.gaussian_preview)

for sample_dir in sample_paths:
    main()
