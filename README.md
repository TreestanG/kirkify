# kirkify

An image processing project I made based on [a video](https://www.youtube.com/shorts/MeFi68a2pP8) I saw, that "kirkifies" any input image. It converts the image CIELAB in order to make color matching more accurate for human perception. Then, it divides pixels into bins by color, sorts by spatial coords, then maps source to target pixels that have similar colors and relative positions. Once mapped, it creates a path for each pixel to go to new position and renders each frame into a final video

### Usage
```
uv run main.py path/to/image.png
```
This works with anything though really, just replace kirk.png with something else


![kirkify](https://github.com/user-attachments/assets/cda5d801-0915-47bc-8431-b5b3bdb9243f)
