import numpy as np
from PIL import Image
from lib import load_portrait_crop, to_processed_gray, foreground_mask, floyd_steinberg_serpentine, GRID_W, GRID_H

SCALE = 4  # preview upscale so dots are visible

def render(dot_mask, path):
    img = Image.new("L", (GRID_W, GRID_H), 255)
    arr = np.array(img)
    arr[dot_mask] = 0
    out = Image.fromarray(arr, "L").resize((GRID_W * SCALE, GRID_H * SCALE), Image.NEAREST)
    out.save(path)

crop = load_portrait_crop("source_photo.png")
crop.save("preview_crop.png")

gray = to_processed_gray(crop)
gray.resize((GRID_W * SCALE, GRID_H * SCALE), Image.NEAREST).save("preview_gray.png")

dots_full = floyd_steinberg_serpentine(gray)
render(dots_full, "preview_light_dots.png")
print("light ink coverage:", dots_full.mean())

mask = foreground_mask(crop)
Image.fromarray((mask * 255).astype("uint8")).resize((GRID_W * SCALE, GRID_H * SCALE), Image.NEAREST).save("preview_mask.png")

dots_dark = dots_full & mask
render(dots_dark, "preview_dark_dots.png")
print("dark ink coverage:", dots_dark.mean(), "mask coverage:", mask.mean())
