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
import cv2





### UTILS
# pipe = pipeline("image-segmentation")
# stupid bilinear thing
if not hasattr(Image, 'LINEAR'):
    Image.LINEAR = Image.BILINEAR

### SETTINGS
width_limit =  120
video = VideoFileClip("./videos/wayne.mp4")
total_frames = video.reader.nframes
print(f"there are {total_frames} frames total")
# video = VideoFileClip("./vinsanity.mp4")


#video setup
frames = []
color = True
num_total_frames = 1000

# video = VideoFileClip("./shaunwhite_big_trim.mp4")
# video = VideoFileClip("./snowboard.mp4")

### RAMPS / ASCII STUFF
empty = " "
full = "X"

long_ramp = "MQW#BNqpHERmKdgAGbX8@SDO$PUkwZyF69heT0a&xV%Cs4fY52Lonz3ucJjvItr}{li?1][7<>=)(+*|!/;:-,_~^.'` "
short_ramp = "ME$sj1|-^` "
punc_ramp = "@%#*+=-:. "
ramp_437 = '▓▒░ '
# ramp_437 = '▓▒░ '
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
        # print(average_pixel, math.floor((average_pixel / 255) * len(self.ramp))-1)
        ramp_i = max(math.floor((average_pixel / 255) * len(self.ramp))-1, 0)
        return self.ramp[ramp_i]
    
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
    color = False
    combine_start = time.time()


    if len(masks) == 0:
        masks.append(base_image)

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


    #make frame for video 
    im = Image.new("RGB", (base_width, base_height), "white")
    d = ImageDraw.Draw(im)

    #obscenely inefficient, oh well
    font_path = "../fonts/Courier_New_Bold.ttf"
    font = ImageFont.truetype(font_path, size=int(square_size))



    # combined_ramp_indices = np.where(transparent_mask, base_ramp_indices, mask_ramp_indices)

    # char_array = np.array(list(base_image.ramp))[combined_ramp_indices]

    #start loop
    for y in range(base_height):
        # print("in y")
        if math.floor(y % square_size) == 0:
            betterPixelArr.append([])
            x = 0
            this_row = ''
            this_row_arr = []
            while x < base_width:
                current_pixel = int((y * base_width) + x)
                # print(current_pixel)
                # print(masks[0])
                pixel_to_use = average_pixels_in_area(current_pixel)
                ramp_char = ''
                if masks[0].is_not_transparent(current_pixel):
                    pixel_colors = masks[0].get_pixel(current_pixel)
                    # print(masks[0].get_pixel(current_pixel))
                    # this_row += f'\x1b[38;2;{pixel_colors[0]};{pixel_colors[1]};{pixel_colors[2]}m{masks[0].get_ramp_for_pixel(current_pixel)}\x1b[0m'
                    ramp_char = f'{masks[0].get_ramp_for_pixel(current_pixel)}'
                    # this_row += ramp_char
                else:
                    # print(base_image.get_pixel(current_pixel))
                    pixel_colors = base_image.get_pixel(current_pixel)
                    ramp_char = f'{base_image.get_ramp_for_pixel(current_pixel)}'
                    # this_row += f'\x1b[38;2;{pixel_colors[0]};{pixel_colors[1]};{pixel_colors[2]}m{masks[0].get_ramp_for_pixel(current_pixel)}\x1b[0m'
                    # this_row += base_image.get_ramp_for_pixel(current_pixel)
                this_row_arr.append([(x, y), ramp_char, pixel_colors])
                # d.text((x, y), this_row, fill=pixel_colors, anchor="lt", font=font)
                this_row += ramp_char
                # betterPixelArr[row_num].append(pixels[int((y * width) + x)])
                x += (square_size * (aspect_ratio[1] / aspect_ratio[0]))
                # x += (square_size * (aspect_ratio[1] / aspect_ratio[0])  * 0.8)
            if color:
                for character in this_row_arr:
                    d.text(character[0], character[1], fill=character[2], anchor="lt", font=font)
            else:
                d.text((0, y), this_row, fill=(0, 0, 0, 255), anchor="lt", font=font)
            row_num += 1
                # for character in this_row_arr:
                #     d.text(character[0], character[1], fill="black", anchor="lt", font=font)
            # print(this_row)
        # if color is False:
        #     
    cv2_frame = np.array(im)
    frames.append(cv2_frame)
    im.close()
    print(f"--- Combination Took {(time.time() - combine_start)} seconds")
    # print(cv2_frame.shape)
    # print(base_width, base_height)



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
ins.load_model("../models/pointrend_resnet50.pkl", detection_speed = 'fast')
target_classes = ins.select_target_classes(person=True)


# segment_image.segmentImage("path_to_image", output_image_name = "output_image_path")


frame_number = 0
# last_start = time.time()
for f in video.iter_frames():
    frame_number += 1
    start_time = time.time()
    # print("image arr", len(f), len(f[0]))
    if frame_number < 1005 and frame_number > 900:# total_frames:
        print("Frame #",frame_number)
        # print(f.shape)
        image_path = f"./image_holding/{frame_number}.png"
        img = Image.fromarray(f).convert('RGBA')
        img.save(image_path)
        # print(target_classes)
        target_classes = ins.select_target_classes(person=True)
        results, output = ins.segmentImage(image_path, show_bboxes=True, segment_target_classes= target_classes, output_image_name=f"./image_holding/{frame_number}_output_normal_person.png")
        print(f"--- Segmentation Took {(time.time() - start_time)} seconds")
        og_person_mask = results['masks']
        person_mask = flatten_multiple_masks(results['masks'])
        print(len(person_mask.shape))



        target_classes = ins.select_target_classes(bicycle=True)
        results, output = ins.segmentImage(image_path, show_bboxes=True, segment_target_classes= target_classes, output_image_name=f"./image_holding/{frame_number}_output_normal_snowboard.png")
        snowboard_mask = flatten_multiple_masks(results['masks'])
        # snowboard_mask = results['masks']
        og_snowboward_mask = results['masks']



        # if (len(p))
        base_image = BaseImage(img, 'base', punc_ramp)
        if len(person_mask.shape) == 3:
            constructed_mask = construct_image_from_arr(f, person_mask)
            masked_image = MaskImage(constructed_mask, 'base', ramp_437)
            # print(constructed_mask)
            # constructed_mask.show()
            # cv2.waitKey(0)
            combine_images(base_image, [masked_image], (16,9))
        else:
            combine_images(base_image, [], (16,9))


        print(f"--- Image Took {(time.time() - start_time)} seconds --- {len(person_mask.shape)}")


#video stuff:
video_dim = (base_image.width, base_image.height)
# fps = 25
vidwriter = cv2.VideoWriter(f"wayne_{int(video.fps)}_grid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), video.fps, video_dim)
print(f"There are {len(frames)} frames")
for frame in frames:
    print(frame.shape)
    frame = cv2.resize(frame,video_dim)
    print(frame.shape)
    vidwriter.write(frame)
vidwriter.release()
print(f"--- Video Took {(time.time() - start_time)} seconds: (width_limit {width_limit}) (num_total_frames {num_total_frames})")