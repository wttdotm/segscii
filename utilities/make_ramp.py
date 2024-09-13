
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
font = ImageFont.truetype("../fonts/Courier_New_Bold.ttf", size=45)

print("This is the name of the program:", sys.argv[0])

print("Argument List:", str(sys.argv))



class Char:
     def __init__(self, char, num_black):
        self.char = char
        self.num_black = num_black

def string_to_ramp(string):
    all_ramps = []
    for char in string:
        # print(f"cache_ramp_maps: {char}")
        char_map_pil = Image.new('L', (40, 40), color="white")
        d = ImageDraw.Draw(char_map_pil)
        d.text((2, 20), char, fill="black", anchor="lm", font=font)
        char_map_cv2 = np.array(char_map_pil)
        num_black = 1600 - cv2.countNonZero(char_map_cv2)
        # print(char, num_black)
        letter_obj = Char(char, num_black)
        all_ramps.append(letter_obj)
        # cv2.imshow(char, char_map_cv2)
        # cv2.waitKey(0)
    sorted_ramps = sorted(all_ramps, key=lambda x:(1000 - x.num_black))
    ramp_joined = ''
    for char in sorted_ramps:
        ramp_joined += char.char
    return ramp_joined
    

print(string_to_ramp(sys.argv[1]))