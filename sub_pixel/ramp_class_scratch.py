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
long_ramp = "MQW#BNHERmKdAGbX8SDOPUkwZF69heT0a&xV%Cs4fY52Lonz3ucJvItr}{li?1][7<>=)(+*|!/-,_~.'` "
short_ramp = "ME$sj1|-^` "
punc_ramp = "@%#*+=-:. "
ramp_437 = '▓▒░ '


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

    # regular ramp funciton
        # take image array
        # make a new array
        # loop through the image
            # for each pixel, average the rgb values
            # append that value to the new array
        # average teh brightness values of that array
        # return whichever char is x bright
    def get_char_region_basic(self, region):
        new_array = []
        for y in range(len(region)):
            for x in range(len(region[y])):
                # i have no idea why I Dont have to divide by 3 here
                color_avg = region[y][x][0] + region[y][x][1] + region[y][x][2]
                new_array.append(color_avg)
        average_brightness = np.average(new_array)
        char = self.get_char_basic(average_brightness)
        # print(f"avg bright: {average_brightness} | char: {char}")
        return char

    # cache maps function
    def cache_ramp_maps(self):
        # Take a width
        for char in self.ramp_string:
            print(f"cache_ramp_maps: {char}")
            char_map_pil = Image.new('RGB', (char_target_width, char_target_height), color="white")
            d = ImageDraw.Draw(char_map_pil)
            d.text((0, 0), char, fill="black", anchor="lt", font=font)
            char_map_cv2 = np.array(char_map_pil)[:, :, ::-1].copy()
            # char_map_cv2 = np.array(char_map_pil.getdata())

            # I could probably optimize this by flattening this brihgtness
            self.brightness_dict[char] = char_map_cv2

        # Iterate through the string of the map
        # Create a PIL image of that character with the defined font and width
        # Save it as a 2d array in a dict with the key of its original character


    def get_char_region_subpixel(self, region)
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


long = Ramp(long_ramp)
long.cache_ramp_maps()

    # def regular_ramp(self, img):



    
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
