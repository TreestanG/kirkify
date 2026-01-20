import argparse
from kirk.io import same_res, to_lab_pixels, match_by_bins, save_animation_video
import os

def main():
    parser = argparse.ArgumentParser(description="Kirkify an image with pixel migration animation.")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("--method", choices=["linear", "fluid"], default="linear", help="Simulation method (default: linear)")
    parser.add_argument("--output", default="kirkify.mp4", help="Output video filename")
    
    args = parser.parse_args()
    
    input_path = args.input
    kirk_path = "images/kirk.png"
    
    if not os.path.exists(input_path) or not os.path.exists(kirk_path):
        print(f"Make sure {input_path} and {kirk_path} exist.")
        return

    source_img, target_img = same_res(input_path, kirk_path)
    
    source_pixels = to_lab_pixels(source_img)
    target_pixels = to_lab_pixels(target_img)
    
    shape = (target_img.height, target_img.width, 3)
    matched_pixels, src_order, tgt_order = match_by_bins(source_pixels, target_pixels, shape)
    
    save_animation_video(
        source_pixels, 
        src_order, 
        tgt_order, 
        shape, 
        filename=args.output, 
        fps=30, 
        duration=5, 
        start_pause=1.5,
        end_pause=2.0,
        method=args.method
    )

if __name__ == "__main__":
    main()
