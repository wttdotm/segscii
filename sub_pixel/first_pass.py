import os 
import math
import itertools as it
import numpy as np
import time
import subprocess
import os 
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
# from transformers import pipeline
import time
import cv2

### EXAMPLE RAMPS / ASCII STUFF
empty = " "
full = "X"
long_ramp = "MQW#BNqpHERmKdgAGbX8@SDO$PUkwZyF69heT0a&xV%Cs4fY52Lonz3ucJjvItr}{li?1][7<>=)(+*|!/;:-,_~^.'` "
short_ramp = "ME$sj1|-^` "
punc_ramp = "@%#*+=-:. "
ramp_437 = '▓▒░ '


### SETTINGS
width_limit =  190





# PSEUDOCODE PLANNING

# global stuff
# what font are we using? for courier, it looks like it's basically best as a 1:1 size, which is fine
# Looks like courier font size : actual pixel size = 8 : 5

# flatten color/brightness of image func
    # take image
    # make new flattened_brightness array
    # iterate through image
        # average the RBG values of that image / array
        # push that value to the new 


# Convert image into ascii basic func

    # Step 1: basics
        # Take source image, i_width, i_height, and target horizontal rez
        # Char width is image_width / target horizontal res
        # Char height is char width * aspect tatio
        # Vertical res is image_height / char_height
        # char_width x char_height is also the pixel unit
font_path = "../fonts/Courier_New_Bold.ttf"
font_w_h_aspect_ratio = 1 #true for courier at least, based on some tests
font_size_to_pixel_ratio = 8/5 # also true for courier based on some tests

image_path = "./test_tree.png"
horizontal_resolution = 3 # can be changed to whatever. Bigger is probably better.

# idk if PIL is the way to go for this
# img_to_convert = Image.open(image_path)

img_to_convert = cv2.imread(image_path)
i_width = img_to_convert.shape[1]
i_height = img_to_convert.shape[0]
print(f"i_width: {i_width}")
print(f"i_height: {i_height}")
print(f"horizontal_resolution: {horizontal_resolution}")

# how many pixels does each char need to be wide
char_target_width = int(i_width / horizontal_resolution)
char_target_height = int(char_target_width * font_w_h_aspect_ratio)
print(f"char_target_width: {char_target_width}")
print(f"char_target_height: {char_target_height}")

# what size font do we need to feed into PIL achieve that? apply aspect ratio
char_display_width = char_target_width * font_size_to_pixel_ratio
char_display_height = char_display_width * font_w_h_aspect_ratio
print(f"char_display_width: {char_display_width}")
print(f"char_display_height: {char_display_height}")

#so we can fit horizontal_resolution chars across, how many can we fit down?
vertical_resolution = int(i_height / char_target_height)
print(f"ASCII dimensions: {horizontal_resolution} x {vertical_resolution}") 

#make font
font = ImageFont.truetype(font_path, size=char_display_width)
    
    # Step 2: Memoization of sub-pixel maps
        # Ramp class
            # Take a ramp and make it into a class
                # ramp string
                # empty dict
                # emtpy dict value dimensions
class Ramp: 
    def __init__(self, string):
        self.ramp_string = string
        self.brightness_dict = {}
        self.char_dimensions = {"x":0, "y":0}

    def get_char_basic(self, num):
        return self.ramp_string[math.floor((num / 255) * len(self.ramp_string))-1]

    def get_char_region_basic(self, region):
        
long = Ramp(long_ramp)

    # def regular_ramp(self, img):


            # cache maps function
                # Take a width
                # Make a dict
                # Iterate through the string of the map
                # Create a PIL image of that character with the defined font and width
                # Save it as a 2d array in a dict with the key of its original character
            # regular ramp funciton
                # take image array
                # make a new array
                # loop through the image
                    # for each pixel, average the rgb values
                    # append that value to the new array
                # average teh brightness values of that array
                # return whichever char is x bright
            # sub-pixel ramp function
                # take image array
                # confirm that iamge array is same dimensions as the dict values
                # create a holder variable for character
                # create a holder variable for lowest_difference
                # iterate through the dict
                    # for each char_img
                        # make a holder difference value
                        # iterate through the char_img
                            # for each pixel
                                # flatten the brightness of the matching img_pixel 
                                # compare it to the brightness of the char_pixel
                                # add that # to the holder value
                            # at the end
                                # check the holder value
                                # if it's lower than the curret lowest_difference
                                # replace the holder character witht he current char (/key)
                                # replace the holder lowest_diff with the new lowest_diff

                # 
    
    # Step 3: Start making new image
        # Make new image of size (char_width * horizontal_rez, char_height * vertical_rez)
        # Iterate through original image
            # for each row of vertical res:
                # for each col of horizontal res
                    # grab the pixels from (currentX, currentY) and (currentX + char_width, currY + char_width)
                        # if using no sub-pixel
                            # get the result of sending it to ramp.get_regular
                        # if using sub-pixel
                            # get the result of sending it to ramp.get_subpixel
                    # Add the resulting character to the resulting image at the appropriate coordinates

ascii_image = Image.new("RGB", (int(char_target_width * horizontal_resolution), int(char_target_height * vertical_resolution)), "white")

print(img_to_convert)

print("should be height:", len(img_to_convert))
print("should be width:", len(img_to_convert[0]))
for y in range(vertical_resolution):
    for x in range(horizontal_resolution):
        print(y, x)
        current_y = y * char_target_height
        current_x = x * char_target_width
        subregion = img_to_convert[current_y:current_y+char_target_height, current_x:current_x+char_target_width]
        print(subregion)
        # pastable_image = Image.fromarray(np.uint8(subregion)).convert('RGB')
        # pastable_image.show(title=f"y{y}_x{x}")
        # cv2.imshow(f"subregion y:{current_y} x{current_x}", subregion)
        # cv2.waitKey(0)
print(long.ramp_string)

rampTest = ""
for i in range(255):
    rampTest += long.get_char_basic(i)

print(rampTest)

# print("should be height:", len(img_to_convert[0][0]))
# print(len(img_to_convert[0][0]))

# for x in len(img_to_convert)
# print(len(img_to_convert[1]))
# print(len(img_to_convert[2]))

# PIL objects are annoying
# this will make the pixels easier to work with
# betterPixelArr = []


    # Step 4: Save image
        # done?

        # 
        #


    # Step 2: image -> pixel regions
        # Iterate through the image, 
# Vertical res is based on char height (char width * aspect ratio) and 
# With target horizontal res, find out how many rows we can make (horizontal res))
# 







# ### CLASSES
# class MaskImage:
#     def __init__(self, masked_image, type_of_img = 'mask',  ramp = '@%#*+=-:. '):
#         self.type = type_of_img
#         self.image = masked_image
#         self.width = masked_image.width
#         self.height = masked_image.height
#         self.ramp = ramp
#         self.pixels = masked_image.getdata()
    
#     def is_transparent(self, pixel):
#         return self.pixels[pixel][3] == 0

#     def is_not_transparent(self, pixel):
#         # print(self.pixels[pixel][3], self.pixels[pixel][3] > 0)
#         return self.pixels[pixel][3] > 0
    
#     def get_ramp_for_pixel(self, pixel):
#         average_pixel = (self.pixels[pixel][0] + self.pixels[pixel][1] + self.pixels[pixel][2])/3
#         return self.ramp[math.floor((average_pixel / 255) * len(self.ramp))-1]
    
#     def get_pixel(self, pixel):
#         return self.pixels[pixel]

# class BaseImage:
#     def __init__(self, image, type_of_img = 'base', ramp = '▓▒░ '):
#         self.type = type_of_img
#         self.image = image
#         self.width = image.width
#         self.height = image.height
#         self.ramp = ramp
#         self.pixels = image.getdata()

#     def is_transparent(self, pixel):
#         return self.pixels[pixel][3] == 0

#     def is_not_transparent(self, pixel):
#         return self.pixels[pixel][3] > 0
    
#     def get_ramp_for_pixel(self, pixel):
#         average_pixel = (self.pixels[pixel][0] + self.pixels[pixel][1] + self.pixels[pixel][2])/3
#         return self.ramp[math.floor((average_pixel / 255) * len(self.ramp))-1]

#     def get_pixel(self, pixel):
#         return self.pixels[pixel]

# ### MAIN FUNCTIONS

# def average_pixels_in_area(pixel):
#     return pixel

# def combine_images(base_image, masks, aspect_ratio):
#     #with the base image
#     #get wdith and height
#     base_width = base_image.width
#     base_height = base_image.height

#     # find how big squares will be
#     # the finder limit is how many characters we can reasonably fit across in the finder
#     square_size = base_width / width_limit
#     # get a pixel array of it all
#     pixels = base_image.image.getdata()

#     # PIL objects are annoying
#     # this will make the pixels easier to work with
#     betterPixelArr = []
#     row_num = 0

#     end_string_arr = []

#     # iterate by row
#         # in each row
#             # make a string
#             # every X pixels
#                 # get the ramp to character to append to the string, which depends on
#                 # 
#                 # move over x pixels
#     for y in range(base_height):
#         # print("in y")
#         if math.floor(y % square_size) == 0:
#             betterPixelArr.append([])
#             x = 0
#             this_row = ''
#             while x < base_width:
#                 current_pixel = int((y * base_width) + x)
#                 # print(current_pixel)
#                 # print(masks[0])
#                 pixel_to_use = average_pixels_in_area(current_pixel)
#                 if masks[0].is_not_transparent(current_pixel):
#                     pixel_colors = masks[0].get_pixel(current_pixel)
#                     # print(masks[0].get_pixel(current_pixel))
#                     # this_row += f'\x1b[38;2;{pixel_colors[0]};{pixel_colors[1]};{pixel_colors[2]}m{masks[0].get_ramp_for_pixel(current_pixel)}\x1b[0m'
#                     this_row += f'{masks[0].get_ramp_for_pixel(current_pixel)}'
#                 else:
#                     # print(base_image.get_pixel(current_pixel))
#                     pixel_colors = base_image.get_pixel(current_pixel)
#                     this_row += f'{masks[0].get_ramp_for_pixel(current_pixel)}'
#                     # this_row += f'\x1b[38;2;{pixel_colors[0]};{pixel_colors[1]};{pixel_colors[2]}m{masks[0].get_ramp_for_pixel(current_pixel)}\x1b[0m'
#                     # this_row += base_image.get_ramp_for_pixel(current_pixel)

#                 # betterPixelArr[row_num].append(pixels[int((y * width) + x)])
#                 x += (square_size * (aspect_ratio[1] / aspect_ratio[0])  * 0.8)
#             row_num += 1
#             print(this_row)


# def construct_image_from_arr(base_image_arr, mask_arr, height_first = True):

#     # Convert inputs to NumPy arrays if they aren't already
#     # This ensures we can use NumPy operations on our inputs
#     base_image_arr = np.array(base_image_arr)
#     mask_arr = np.array(mask_arr)

#     # Create a 4-channel RGBA array filled with zeros
#     # The shape is (height, width, 4), where 4 represents RGBA channels
#     # dtype=np.uint8 ensures each value is an 8-bit unsigned integer (0-255)
#     rgba_arr = np.zeros((*base_image_arr.shape[:2], 4), dtype=np.uint8)

#     # Fill RGB channels from base_image_arr
#     # [:,:,:3] selects all rows, all columns, and the first 3 channels (RGB)
#     # This copies the RGB data from base_image_arr to our new rgba_arr
#     rgba_arr[:,:,:3] = base_image_arr[:,:,:3]

#     # Set alpha channel based on mask_arr
#     # np.where(condition, x, y) returns x where condition is True, else y
#     # mask_arr[:,:,0] is True where the mask is applied (assuming mask is boolean)
#     # This sets alpha to 255 (fully opaque) where mask is True, 0 (transparent) elsewhere
#     rgba_arr[:,:,3] = np.where(mask_arr[:,:,0], 255, 0)

#     # Create and return PIL Image
#     # fromarray converts our NumPy array back to a PIL Image object
#     return Image.fromarray(rgba_arr)

# def flatten_multiple_masks(arr):
#     merged_mask = np.any(arr, axis=-1)
#     return np.expand_dims(merged_mask, axis=-1)

# ### WHAT GETS RUN


# import pixellib
# from pixellib.torchbackend.instance import instanceSegmentation
# ins = instanceSegmentation()

# # segment_image = instance_segmentation()
# ins.load_model("../models/pointrend_resnet50.pkl")
# target_classes = ins.select_target_classes(person=True)


# # segment_image.segmentImage("path_to_image", output_image_name = "output_image_path")

# frame_number = 0
# # last_start = time.time()
# for f in video.iter_frames():
#     frame_number += 1
#     start_time = time.time()
#     # print("image arr", len(f), len(f[0]))
#     if frame_number > 1:
#         print("Frame #",frame_number)
#         # print(f.shape)
#         image_path = f"./image_holding/{frame_number}.png"
#         img = Image.fromarray(f).convert('RGBA')
#         img.save(image_path)
#         # print(target_classes)
#         target_classes = ins.select_target_classes(person=True)
#         results, output = ins.segmentImage(image_path, show_bboxes=True, segment_target_classes= target_classes, output_image_name=f"./image_holding/{frame_number}_output_normal_person.png")
#         og_person_mask = results['masks']
#         person_mask = flatten_multiple_masks(results['masks'])



#         target_classes = ins.select_target_classes(snowboard=True)
#         results, output = ins.segmentImage(image_path, show_bboxes=True, segment_target_classes= target_classes, output_image_name=f"./image_holding/{frame_number}_output_normal_snowboard.png")
#         snowboard_mask = flatten_multiple_masks(results['masks'])
#         # snowboard_mask = results['masks']
#         og_snowboward_mask = results['masks']
#         if len(og_snowboward_mask) == 0 and len(og_person_mask) == 0:
#             combined_masks = np.full(f.shape, False)
#         elif len(og_snowboward_mask) == 0:
#             combined_masks = person_mask
#         elif len(og_person_mask) == 0:
#             combined_masks = snowboard_mask
#         elif person_mask.shape == snowboard_mask.shape:
#             combined_masks = np.where(snowboard_mask != False, snowboard_mask, person_mask)
#             # print(snowboard_mask.shape, person_mask.shape)
#         # if person_mask.shape == snowboard_mask.shape:
#         #     combined_masks = np.where(snowboard_mask != False, snowboard_mask, person_mask)


#         # print(results)
#         # print(len(results['masks']))
#         # constructed_mask = construct_image_from_arr(f, results['masks'])
#         constructed_mask = construct_image_from_arr(f, combined_masks)
#         # constructed_mask.show()

#         base_image = BaseImage(img, 'base', long_ramp)
#         masked_image = MaskImage(constructed_mask, 'base', ramp_437)
#         combine_images(base_image, [masked_image], (16,9))
#         print("--- Image Took %s seconds ---" % (time.time() - start_time))

