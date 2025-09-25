from PIL import Image

def open_rgb(p: str) -> Image.Image:
    return Image.open(p).convert("RGB")