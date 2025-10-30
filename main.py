import os
from PIL import Image
import numpy as np
from collections import Counter

# ==========================================================
# CONFIGURATION
# ==========================================================
BUILD_LENGTH = 50  # Number of tiles horizontally
BUILD_WIDTH = 50   # Number of tiles vertically
BLOCK_TYPES = ["Wool", "Concrete", "Terracotta", "Wood", "Stone"] # Available types: ["Wool", "Concrete", "Terracotta", "Wood", "Stone"]

SOURCE_IMAGE = "image.png"

TEXTURE_DIR = "Textures"  # Base directory
SAVE_RESIZED = False      # Whether to save the resized image

# ==========================================================
# STEP 1 — LOAD TEXTURE COLOURS
# ==========================================================
def load_texture_colours(base_dir, block_types):
    """
    Load textures from selected subfolders (specified in block_types list)
    and compute their average RGB colour.
    """
    subfolders = block_types

    if not subfolders:
        raise ValueError("No texture types selected! Add types to BLOCK_TYPES.")

    colour_dict = {}
    for subfolder in subfolders:
        folder_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(folder_path):
            print(f"Warning: folder '{folder_path}' not found, skipping.")
            continue

        print(f"Loading textures from: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".png"):
                path = os.path.join(folder_path, filename)
                pixels = np.array(Image.open(path).convert("RGB"))
                avg_colour = pixels.mean(axis=(0, 1))
                texture_name = filename[:-4]
                internal_key = f"{subfolder}::{texture_name}"
                colour_dict[internal_key] = avg_colour

    if not colour_dict:
        raise RuntimeError("No textures loaded! Check your folder paths and selections.")
    return colour_dict


# ==========================================================
# STEP 2 — RESIZE INPUT IMAGE
# ==========================================================
def resize_image(path, width, height, save=False):
    img = Image.open(path).convert("RGBA")  # keep alpha channel
    resized = img.resize((width, height), Image.Resampling.LANCZOS)
    if save:
        resized.save("resized.png")
    return np.array(resized)  # shape (H, W, 4)


# ==========================================================
# STEP 3 — MAP IMAGE PIXELS TO NEAREST TEXTURE COLOUR
# ==========================================================
def match_texture_layout(image_pixels, colour_dict):
    """
    Match non-transparent pixels to nearest texture colour.
    Transparent pixels will be marked as '__EMPTY__'.
    """
    texture_keys = np.array(list(colour_dict.keys()))
    texture_values = np.stack(list(colour_dict.values()))

    rows, cols, _ = image_pixels.shape
    layout = np.empty((rows, cols), dtype=object)

    print("Matching pixels to textures (ignoring transparent pixels)...")

    rgb_pixels = image_pixels[..., :3].astype(float)
    alpha = image_pixels[..., 3]

    pixels_flat = rgb_pixels.reshape(-1, 3)
    alpha_flat = alpha.flatten()

    # Compute distances only for opaque pixels
    opaque_mask = alpha_flat > 0
    opaque_pixels = pixels_flat[opaque_mask]

    diff = opaque_pixels[:, None, :] - texture_values[None, :, :]
    dists = np.sum(diff ** 2, axis=2)
    best_indices = np.argmin(dists, axis=1)
    best_keys = texture_keys[best_indices]

    layout_flat = np.full(alpha_flat.shape, "__EMPTY__", dtype=object)
    layout_flat[opaque_mask] = best_keys
    layout = layout_flat.reshape(rows, cols)

    return layout


# ==========================================================
# STEP 4 — BUILD MOSAIC IMAGE FROM LAYOUT
# ==========================================================
def build_mosaic(layout, base_dir):
    unique_keys = [k for k in np.unique(layout) if k != "__EMPTY__"]
    tile_dict = {}

    for key in unique_keys:
        subfolder, tex_name = key.split("::")
        tile_path = os.path.join(base_dir, subfolder, f"{tex_name}.png")
        tile_dict[key] = Image.open(tile_path).convert("RGB")

    tile_w, tile_h = next(iter(tile_dict.values())).size
    rows, cols = layout.shape
    mosaic_img = Image.new("RGBA", (cols * tile_w, rows * tile_h), (0, 0, 0, 0))

    print("Building mosaic (transparent areas left blank)...")
    for i in range(rows):
        for j in range(cols):
            key = layout[i, j]
            if key == "__EMPTY__":
                continue  # skip transparent pixel
            tile = tile_dict[key]
            mosaic_img.paste(tile, (j * tile_w, i * tile_h))
        if i % 10 == 0:
            print(f"  Row {i}/{rows} complete")

    mosaic_img.save("mosaic.png")
    print("Mosaic saved as mosaic.png")
    return mosaic_img


# ==========================================================
# STEP 5 — COUNT REQUIRED TEXTURES
# ==========================================================
def count_textures(layout):
    counts = Counter()
    for row in layout:
        for key in row:
            if key == "__EMPTY__":
                continue
            _, tex_name = key.split("::")
            counts[tex_name] += 1

    print("\nTexture count summary:")
    for texture, count in sorted(counts.items()):
        stacks = count // 64
        remainder = count % 64
        if stacks > 0:
            stack_str = f"{stacks} stack" if stacks == 1 else f"{stacks} stacks"
            output = f"{stack_str} + {remainder}" if remainder > 0 else stack_str
        else:
            output = f"{remainder}"
        print(f"{texture}: {output}")

    with open("texture_counts.txt", "w") as f:
        for texture, count in sorted(counts.items()):
            stacks = count // 64
            remainder = count % 64
            if stacks > 0:
                stack_str = f"{stacks} stack" if stacks == 1 else f"{stacks} stacks"
                output = f"{stack_str} + {remainder}" if remainder > 0 else stack_str
            else:
                output = f"{remainder}"
            f.write(f"{texture}: {output}\n")

    return counts


# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("Processing textures...")

    texture_colours = load_texture_colours(TEXTURE_DIR, BLOCK_TYPES)
    resized_pixels = resize_image(SOURCE_IMAGE, BUILD_LENGTH, BUILD_WIDTH, SAVE_RESIZED)
    texture_layout = match_texture_layout(resized_pixels, texture_colours)
    build_mosaic(texture_layout, TEXTURE_DIR)
    count_textures(texture_layout)

    print("\nDone! Mosaic and texture counts generated.")
