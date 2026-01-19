from kirk.io import same_res, to_lab_pixels, from_lab_pixels, match_by_bins, save_animation_video
import os, sys
import numpy as np

def main():
    input_path = sys.argv[1]
    kirk_path = "images/kirk.png"
    
    if not os.path.exists(os.path.join(input_path)) or not os.path.exists(kirk_path):
        print("Make sure images/input.png and images/kirk.png exist.")
        return

    source_img, target_img = same_res(os.path.join(input_path), os.path.join(kirk_path))
    
    source_pixels = to_lab_pixels(source_img)
    target_pixels = to_lab_pixels(target_img)
    
    shape = (target_img.height, target_img.width, 3)
    matched_pixels, src_order, tgt_order = match_by_bins(source_pixels, target_pixels, shape)
    
    save_animation_video(
        source_pixels, 
        src_order, 
        tgt_order, 
        shape, 
        filename="kirkify.mp4", 
        fps=30, 
        duration=5,
        start_pause=1.5,
        end_pause=2.0
    )

if __name__ == "__main__":
    main()
