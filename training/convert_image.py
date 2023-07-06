from PIL import Image

def convert_image(input_image_path, output_image_path):
    img = Image.open(input_image_path)
    img = img.convert("RGB")

    d = img.getdata()

    new_image = []
    for item in d:
        # change all non-black (also shades of grays)
        # pixels to white
        if item != (0, 0, 0):
            new_image.append((255, 255, 255))
        else:
            new_image.append(item)
            
    img.putdata(new_image)
    img.save(output_image_path)

i = 3
base = "/home/tanujthakkar/ash/PRG-Hoop-Recognition/assets/hoops/hoop8.png"
convert_image(base, "output.jpg")
