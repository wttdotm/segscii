import os 
import math
import itertools as it
import numpy as np
import time
import subprocess
import os 
from PIL import Image, ImageDraw, ImageFont


bad_apple_masked = Image.open('frame_00271_masked.png').convert('RGBA')
bad_apple_nonmasked = Image.open('frame_00271_nonmasked.png').convert('RGBA')
width_limit =  60

empty = " "
full = "X"

long_ramp = "MQW#BNqpHERmKdgAGbX8@SDO$PUkwZyF69heT0a&xV%Cs4fY52Lonz3ucJjvItr}{li?1][7<>=)(+*|!/;:-,_~^.'` "
short_ramp = "ME$sj1|-^` "
punc_ramp = "@%#*+=-:. "
ramp_437 = '▓▒░ '

# some class stuff
class MaskImage:
    def __init__(self, masked_image, type_of_img = 'mask',  ramp = '@%#*+=-:. '):
        self.type = type_of_img
        self.image = masked_image
        self.width = masked_image.width
        self.height = masked_image.height
        self.ramp = ramp
        self.pixels = masked_image.getdata()
    
    def is_transparent(self, pixel):
        return self.pixels[pixel][3] == 0

    def is_not_transparent(self, pixel):
        # print(self.pixels[pixel][3], self.pixels[pixel][3] > 0)
        return self.pixels[pixel][3] > 0
    
    def get_ramp_for_pixel(self, pixel):
        average_pixel = (self.pixels[pixel][0] + self.pixels[pixel][1] + self.pixels[pixel][2])/3
        return self.ramp[math.floor((average_pixel / 255) * len(self.ramp))-1]

class BaseImage:
    def __init__(self, image, type_of_img = 'base', ramp = '▓▒░ '):
        self.type = type_of_img
        self.image = image
        self.width = image.width
        self.height = image.height
        self.ramp = ramp
        self.pixels = image.getdata()

    def is_transparent(self, pixel):
        return self.pixels[pixel][3] == 0

    def is_not_transparent(self, pixel):
        return self.pixels[pixel][3] > 0
    
    def get_ramp_for_pixel(self, pixel):
        average_pixel = (self.pixels[pixel][0] + self.pixels[pixel][1] + self.pixels[pixel][2])/3
        return self.ramp[math.floor((average_pixel / 255) * len(self.ramp))-1]



def average_pixels_in_area(pixel):
    return pixel

def combine_images(base_image, masks, aspect_ratio):
    #with the base image
    #get wdith and height
    base_width = base_image.width
    base_height = base_image.height

    # find how big squares will be
    # the finder limit is how many characters we can reasonably fit across in the finder
    square_size = base_width / width_limit
    # get a pixel array of it all
    pixels = base_image.image.getdata()

    # PIL objects are annoying
    # this will make the pixels easier to work with
    betterPixelArr = []
    row_num = 0

    end_string_arr = []

    # iterate by row
        # in each row
            # make a string
            # every X pixels
                # get the ramp to character to append to the string, which depends on
                # 
                # move over x pixels
    for y in range(base_height):
        # print("in y")
        if math.floor(y % square_size) == 0:
            betterPixelArr.append([])
            x = 0
            this_row = ''
            while x < base_width:
                current_pixel = int((y * base_width) + x)
                # print(current_pixel)
                # print(masks[0])
                pixel_to_use = average_pixels_in_area(current_pixel)
                # print(current_pixel)
                if masks[0].is_not_transparent(current_pixel):
                    this_row += masks[0].get_ramp_for_pixel(current_pixel)
                else:
                    this_row += base_image.get_ramp_for_pixel(current_pixel)

                # betterPixelArr[row_num].append(pixels[int((y * width) + x)])
                x += (square_size * (aspect_ratio[1] / aspect_ratio[0])  * 0.8)
            row_num += 1
            print(this_row)




bad_apple_masked = MaskImage(bad_apple_masked, 'mask', long_ramp)
bad_apple_nonmasked = BaseImage(bad_apple_nonmasked, 'base', ramp_437)

from moviepy.editor import VideoFileClip
from transformers import pipeline

# pipe = pipeline("image-segmentation",  model="CIDAS/clipseg-rd64-refined")
pipe = pipeline("image-segmentation")


# video = VideoFileClip("vinsanity.mp4")
video = VideoFileClip("../infinite-video-fall-2023/08_motion_and_segmentation/vinsanity.mp4")

frame_number = 0
for f in video.iter_frames():
    frame_number += 1
    if frame_number > 2600:
        img = Image.fromarray(f).convert('RGBA')
        results = pipe(img)
        non_masked_img = BaseImage(img, 'base')
        masked_image = MaskImage(img, 'mask')
        # print(masked_image.pixels)
        # img.save(non_masked_output_path)
        has_person = False
        mask = False
        print(results)
        for i, r in enumerate(results):
            if r["label"] == "person":
                mask = r["mask"]
        if mask is not False:
            print("foundPerson")
            img.putalpha(mask)
            masked_image = MaskImage(img, 'mask',long_ramp)
            # print(masked_image.pixels)
            # img.putalpha(r["mask"])
            # print(r)
            combine_images(non_masked_img, [masked_image], (4,3))
        else:
            print("in else")
            combine_images(non_masked_img, [non_masked_img], (4,3))
        # for i, r in enumerate(results):
        #     print(i, r)
        #     # has_person = False
        #     if r["label"] == "person":
        #         print("foundPerson")
        #         has_person = True
        #         img.putalpha(r["mask"])
        #         masked_image = MaskImage(img, 'mask',long_ramp)
        #         # print(masked_image.pixels)
        #         # img.putalpha(r["mask"])
        #         # print(r)
        #         combine_images(non_masked_img, [masked_image], (4,3))
        #         continue
        #     elif has_person is False:
        #         print("in else")
        #         combine_images(non_masked_img, [non_masked_img], (4,3))
        # img.save(masked_output_path)/


# combine_images(bad_apple_nonmasked, [bad_apple_masked], (4,3))