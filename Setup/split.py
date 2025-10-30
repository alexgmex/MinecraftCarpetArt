from PIL import Image

# Input image file
input_file = "all_carpets.png"

# Output names (in grid order, top-left to bottom-right)
carpet_names = [
    "red", "orange", "yellow", "lime",
    "green", "cyan", "lightblue", "blue",
    "purple", "magenta", "pink", "white",
    "lightgrey", "grey", "black", "brown"
]

# Open the source image
img = Image.open(input_file)
width, height = img.size

# Grid dimensions
cols, rows = 4, 4
tile_width = width // cols
tile_height = height // rows

# Split and save each tile
for i, name in enumerate(carpet_names):
    col = i % cols
    row = i // cols
    left = col * tile_width
    upper = row * tile_height
    right = left + tile_width
    lower = upper + tile_height
    tile = img.crop((left, upper, right, lower))
    tile.save(f"{name}.png")

print("Done! All carpet textures have been saved individually.")
