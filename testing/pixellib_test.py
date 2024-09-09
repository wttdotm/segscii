import os 
import math
import itertools as it
import numpy as np
import time
import subprocess
import os 
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
from transformers import pipeline
import time




### UTILS
# pipe = pipeline("image-segmentation")
# stupid bilinear thing
if not hasattr(Image, 'LINEAR'):
    Image.LINEAR = Image.BILINEAR

### SETTINGS
width_limit =  190
# video = VideoFileClip("./vinsanity.mp4")
# video = VideoFileClip("./shaunwhite_big_trim.mp4")
video = VideoFileClip("./snowboard.mp4")

### RAMPS / ASCII STUFF
empty = " "
full = "X"

long_ramp = "MQW#BNqpHERmKdgAGbX8@SDO$PUkwZyF69heT0a&xV%Cs4fY52Lonz3ucJjvItr}{li?1][7<>=)(+*|!/;:-,_~^.'` "
short_ramp = "ME$sj1|-^` "
punc_ramp = "@%#*+=-:. "
ramp_437 = '▓▒░ '

### CLASSES
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
    
    def get_pixel(self, pixel):
        return self.pixels[pixel]

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

    def get_pixel(self, pixel):
        return self.pixels[pixel]

### MAIN FUNCTIONS

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
                if masks[0].is_not_transparent(current_pixel):
                    pixel_colors = masks[0].get_pixel(current_pixel)
                    # print(masks[0].get_pixel(current_pixel))
                    # this_row += f'\x1b[38;2;{pixel_colors[0]};{pixel_colors[1]};{pixel_colors[2]}m{masks[0].get_ramp_for_pixel(current_pixel)}\x1b[0m'
                    this_row += f'{masks[0].get_ramp_for_pixel(current_pixel)}'
                else:
                    # print(base_image.get_pixel(current_pixel))
                    pixel_colors = base_image.get_pixel(current_pixel)
                    this_row += f'{masks[0].get_ramp_for_pixel(current_pixel)}'
                    # this_row += f'\x1b[38;2;{pixel_colors[0]};{pixel_colors[1]};{pixel_colors[2]}m{masks[0].get_ramp_for_pixel(current_pixel)}\x1b[0m'
                    # this_row += base_image.get_ramp_for_pixel(current_pixel)

                # betterPixelArr[row_num].append(pixels[int((y * width) + x)])
                x += (square_size * (aspect_ratio[1] / aspect_ratio[0])  * 0.8)
            row_num += 1
            print(this_row)


def construct_image_from_arr(base_image_arr, mask_arr, height_first = True):

    # Convert inputs to NumPy arrays if they aren't already
    # This ensures we can use NumPy operations on our inputs
    base_image_arr = np.array(base_image_arr)
    mask_arr = np.array(mask_arr)

    # Create a 4-channel RGBA array filled with zeros
    # The shape is (height, width, 4), where 4 represents RGBA channels
    # dtype=np.uint8 ensures each value is an 8-bit unsigned integer (0-255)
    rgba_arr = np.zeros((*base_image_arr.shape[:2], 4), dtype=np.uint8)

    # Fill RGB channels from base_image_arr
    # [:,:,:3] selects all rows, all columns, and the first 3 channels (RGB)
    # This copies the RGB data from base_image_arr to our new rgba_arr
    rgba_arr[:,:,:3] = base_image_arr[:,:,:3]

    # Set alpha channel based on mask_arr
    # np.where(condition, x, y) returns x where condition is True, else y
    # mask_arr[:,:,0] is True where the mask is applied (assuming mask is boolean)
    # This sets alpha to 255 (fully opaque) where mask is True, 0 (transparent) elsewhere
    rgba_arr[:,:,3] = np.where(mask_arr[:,:,0], 255, 0)

    # Create and return PIL Image
    # fromarray converts our NumPy array back to a PIL Image object
    return Image.fromarray(rgba_arr)

def flatten_multiple_masks(arr):
    merged_mask = np.any(arr, axis=-1)
    return np.expand_dims(merged_mask, axis=-1)

### WHAT GETS RUN


import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
ins = instanceSegmentation()

# segment_image = instance_segmentation()
ins.load_model("../models/pointrend_resnet50.pkl")
target_classes = ins.select_target_classes(person=True)


# segment_image.segmentImage("path_to_image", output_image_name = "output_image_path")

frame_number = 0
# last_start = time.time()
for f in video.iter_frames():
    frame_number += 1
    start_time = time.time()
    # print("image arr", len(f), len(f[0]))
    if frame_number > 1:
        print("Frame #",frame_number)
        # print(f.shape)
        image_path = f"./image_holding/{frame_number}.png"
        img = Image.fromarray(f).convert('RGBA')
        img.save(image_path)
        # print(target_classes)
        target_classes = ins.select_target_classes(person=True)
        results, output = ins.segmentImage(image_path, show_bboxes=True, segment_target_classes= target_classes, output_image_name=f"./image_holding/{frame_number}_output_normal_person.png")
        og_person_mask = results['masks']
        person_mask = flatten_multiple_masks(results['masks'])



        target_classes = ins.select_target_classes(snowboard=True)
        results, output = ins.segmentImage(image_path, show_bboxes=True, segment_target_classes= target_classes, output_image_name=f"./image_holding/{frame_number}_output_normal_snowboard.png")
        snowboard_mask = flatten_multiple_masks(results['masks'])
        # snowboard_mask = results['masks']
        og_snowboward_mask = results['masks']
        # print("og snowboard", snowboard_mask)
        # merged_snowboard_mask = np.any(snowboard_mask, axis=-1)
        # snowboard_mask = np.expand_dims(merged_snowboard_mask, axis=-1)

        # print(merged_snowboard_mask)
        # print("shapes", snowboard_mask.shape, person_mask.shape, merged_snowboard_mask.shape)
        # print(snowboard_mask, person_mask)
        # print(len(snowboard_mask.shape), len(person_mask.shape), len(merged_snowboard_mask.shape))
        # if (len(snowboard_mask) == 0 and len(person_mask))


        # if len(snowboard_mask.shape) == 1:
        #     # print("hit snowbaord mask shape len 1")
        #     combined_masks = person_mask
        # elif len(person_mask.shape) == 1:
        #     combined_masks = snowboard_mask
        # elif (snowboard_mask.shape[2] > 1):
        #     masks_arr = []
        #     for i in range(snowboard_mask.shape[2]):
        #         # print("range snowmask shape", i)
        #         # new_mask = np.array([[isinstance(x[i], tuple) for x in row] for row in snowboard_mask])
        #         new_mask = np.array([[[isinstance(x[i], tuple)] for x in row] for row in snowboard_mask])

        #         # print(new_mask)
        #         masks_arr.append(new_mask)
        #     base_mask_arr = masks_arr[0]
        #     for i in range(len(masks_arr)):
        #         # print("range len amsks arr", i)
        #         # print(masks_arr[i].shape)
        #         base_mask_arr = np.where(masks_arr[i] != False, masks_arr[i], base_mask_arr)
        #     snowboard_mask = base_mask_arr
        #     # combined_masks = np.where(snowboard_mask != False, snowboard_mask, person_mask):
        #     # print(snowboard_mask)
        
        if len(og_snowboward_mask) == 0 and len(og_person_mask) == 0:
            combined_masks = np.full(f.shape, False)
        elif len(og_snowboward_mask) == 0:
            combined_masks = person_mask
        elif len(og_person_mask) == 0:
            combined_masks = snowboard_mask
        elif person_mask.shape == snowboard_mask.shape:
            combined_masks = np.where(snowboard_mask != False, snowboard_mask, person_mask)
            # print(snowboard_mask.shape, person_mask.shape)
        # if person_mask.shape == snowboard_mask.shape:
        #     combined_masks = np.where(snowboard_mask != False, snowboard_mask, person_mask)


        # print(results)
        # print(len(results['masks']))
        # constructed_mask = construct_image_from_arr(f, results['masks'])
        constructed_mask = construct_image_from_arr(f, combined_masks)
        # constructed_mask.show()

        base_image = BaseImage(img, 'base', long_ramp)
        masked_image = MaskImage(constructed_mask, 'base', ramp_437)
        combine_images(base_image, [masked_image], (16,9))
        print("--- Image Took %s seconds ---" % (time.time() - start_time))

