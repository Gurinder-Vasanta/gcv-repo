import PIL.Image
import numpy as np
pic = PIL.Image.open('pic1.jpeg')
print(pic.size)
pic_rgb = pic.convert("RGB")
for i in range(4032):
    for j in range(3024):
        print(np.sum(np.array(pic_rgb.getpixel((i,j))))/3)
#rgb_pixel_value = pic_rgb.getpixel((101,105))
#print(rgb_pixel_value)
