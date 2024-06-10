# import numpy as np
# import torch
# from PIL import Image
# from diffusers import DiffusionPipeline
# from diffusers.utils import load_image
# import matplotlib as plt


# def MonocularDepthEstimationMarigold(color_frame,min_dist,max_dist):
#     # Original DDIM version (higher quality)
#     # pipe = DiffusionPipeline.from_pretrained(
#     #     "prs-eth/marigold-v1-0",
#     #     custom_pipeline="marigold_depth_estimation"
#     #     # torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
#     #     # variant="fp16",                           # (optional) Use with `torch_dtype=torch.float16`, to directly load fp16 checkpoint
#     # )

#     # (New) LCM version (faster speed)
#     pipe = DiffusionPipeline.from_pretrained(
#         "prs-eth/marigold-lcm-v1-0",
#         custom_pipeline="marigold_depth_estimation"
#         # torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
#         # variant="fp16",                           # (optional) Use with `torch_dtype=torch.float16`, to directly load fp16 checkpoint
#     )

#     pipe.to("cuda")

#     image = Image.fromarray(color_frame)

#     pipeline_output = pipe(
#         image,                    # Input image.
#         # ----- recommended setting for DDIM version -----
#         # denoising_steps=10,     # (optional) Number of denoising steps of each inference pass. Default: 10.
#         # ensemble_size=10,       # (optional) Number of inference passes in the ensemble. Default: 10.
#         # ------------------------------------------------
        
#         # ----- recommended setting for LCM version ------
#         denoising_steps=4,
#         ensemble_size=5,
#         # -------------------------------------------------
        
#         # processing_res=768,     # (optional) Maximum resolution of processing. If set to 0: will not resize at all. Defaults to 768.
#         # match_input_res=True,   # (optional) Resize depth prediction to match input resolution.
#         # batch_size=0,           # (optional) Inference batch size, no bigger than `num_ensemble`. If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
#         # seed=2024,              # (optional) Random seed can be set to ensure additional reproducibility. Default: None (unseeded). Note: forcing --batch_size 1 helps to increase reproducibility. To ensure full reproducibility, deterministic mode needs to be used.
#         # color_map="Spectral",   # (optional) Colormap used to colorize the depth map. Defaults to "Spectral". Set to `None` to skip colormap generation.
#         # show_progress_bar=True, # (optional) If true, will show progress bars of the inference progress.
#     )

#     depth = np.ndarray = pipeline_output.depth_np                    # Predicted depth map
#     #depth_colored = Image.Image = pipeline_output.depth_colored      # Colorized prediction

#     # Save as uint16 PNG
#     depth_frame = (depth * 65535.0).astype(np.uint16)
#     print(depth_frame)
#     plt.imshow(depth_frame)
#     plt.show()
#     #Image.fromarray(depth_uint16).save("./depth_map.png", mode="I;16")

#     # Save colorized depth map
#     #depth_colored.save("./depth_colored.png")