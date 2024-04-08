from PIL import Image


def convert(p):
    img = Image.open(p)
    img_L = img.convert("L")
    return img_L


def convert_and_save(p, o):
    convert(p).save(o)
