import time
import numpy as np

from PIL import Image


def sliding_window_make_patches(save_path, slide_id, wsi, segmentation):
    start = time.time()
    startpoint = (0, 0)
    tile_size = 1024

    shape = wsi.level_dimensions[0]  # 40X
    share_x = (shape[0]) // tile_size
    share_y = (shape[1]) // tile_size

    for i in range(share_y):
        for j in range(share_x):
            startpoint = (j * 2048, i * 2048)
            image = wsi.read_region(startpoint, 2, (512, 512)).convert("RGB")  # 10X
            image = np.array(image)

            is_zero_present = np.any(np.all(image == [0, 0, 0], axis=2))
            above_threshold_pixels = image > 230
            above_threshold_count = np.sum(above_threshold_pixels)
            under_threshold_pixels = image < 20
            under_threshold_count = np.sum(under_threshold_pixels)
            total_pixels = image.size
            ratio_noise = (under_threshold_count + above_threshold_count) / total_pixels

            if image.sum() != 0 and is_zero_present == False and ratio_noise < 0.5:
                seg_point = (startpoint[0] // 16, startpoint[1] // 16)
                seg_patch = segmentation[
                    seg_point[1] : seg_point[1] + 128, seg_point[0] : seg_point[0] + 128
                ]
                total = seg_patch.size
                nzero = np.count_nonzero(seg_patch)
                image = Image.fromarray(image)

                if nzero >= (total / 2):
                    image.save(
                        f"{save_path}/{slide_id}_x_{startpoint[0]}_y_{startpoint[1]}.png",
                        "BMP",
                    )

        print(f"time : {time.time()-start}")
