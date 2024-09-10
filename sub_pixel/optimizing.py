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
from skimage import metrics


start_time = time.time()


### EXAMPLE RAMPS / ASCII STUFF
empty = " "
full = "X"
# long_ramp = "ABCDEF"
# long_ramp = "XMI/-."
long_ramp = "MQW#BNHERmKdAGbX8SDOPUkwZF69heT0a&xV%Cs4fY52Lonz3ucJvItr}{li?1][7<>=)(+*|!/-,_~.'` "
# long_ramp = "s4fY52Lonz3ucJvItr}{li?1][7<>=)(+*|!/-,_~.'` "
short_ramp = "ME$sj1|-^` "
punc_ramp = "@%#*+=-:. "
ramp_437 = '▓▒░ '



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
# font_path = "../fonts/Courier_New_Bold.ttf"
font_w_h_aspect_ratio = 1 #true for courier at least, based on some tests
font_size_to_pixel_ratio = 8/5 # also true for courier based on some tests

# image_path = "./cursive_test.png"
image_path = "./test_tree.png"
# image_path = "./test_tree2.jpg"
# image_path = "./ABC.png"
# image_path = "./abcdef2.png"
# image_path = "./ABCDEF.png"
horizontal_resolution = 150 # can be changed to whatever. Bigger is probably better.

# idk if PIL is the way to go for this
# img_to_convert = Image.open(image_path)

# img_to_convert = cv2.imread(image_path)
img_to_convert = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


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
        self.brightness_dict_flat = {}
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
        # weights = np.array([0.299, 0.587, 0.114])  # Common weights for perceived brightness
        # region_averaged = np.sum(region * weights, axis=-1, keepdims=True)
        # # region_averaged = np.mean(region, axis=-1, keepdims=True)
        new_array = []
        average_bright = np.mean(region)
        # print("avg bright:", average_bright)
        if average_bright > 200:
            return " "
        for y in range(len(region)):
            for x in range(len(region[y])):
                # i have no idea why I Dont have to divide by 3 here
                color_avg = region[y][x]
                new_array.append(color_avg)
        average_brightness = np.average(new_array)
        char = self.get_char_basic(average_brightness)
        # print(f"avg bright: {average_brightness} | char: {char}")
        return char

        # cache maps function
            # Take a width
            # Make a dict
            # Iterate through the string of the map
            # Create a PIL image of that character with the defined font and width
            # Save it as a 2d array in a dict with the key of its original character
    def cache_ramp_maps(self):
        # Take a width
        for char in self.ramp_string:
            # print(f"cache_ramp_maps: {char}")
            char_map_pil = Image.new('L', (char_target_width, char_target_height), color="white")
            d = ImageDraw.Draw(char_map_pil)
            d.text((0, 0), char, fill="black", anchor="lt", font=font)
            char_map_cv2 = np.array(char_map_pil)
            # char_map_cv2 = np.array(char_map_pil)[:, :, ::-1].copy()
            # char_map_cv2 = np.array(char_map_pil.getdata())

            # I could probably optimize this by flattening this brihgtness
            # self.brightness_dict[char] = char_map_cv2
            
            #ok im gonna do it 
            flattened = np.mean(char_map_cv2, axis=-1, keepdims=True)
            # print(char_map_cv2)
            self.brightness_dict_flat[char] = char_map_cv2
        



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

    def get_char_region_subpixel(self, region):
        # print("\n\n\n\n", region)
        # print("region", region)
        # region_averaged = np.mean(region, axis=-1, keepdims=True)
        # weights = np.array([0.299, 0.587, 0.114])  # Common weights for perceived brightness
        # region_averaged = np.sum(region * weights, axis=-1, keepdims=True)
        best_ssim = 0
        char_holder = ''
        # if total == 200:
        #     cv2.imshow("subregion at 200", region)
        #     cv2.waitKey(0)
        # start difference extremely high
        lowest_difference = float('inf')

        highest_similarity = float('-inf')

        # print("shapes in subpixel:", region.shape, self.brightness_dict["W"].shape)
        # if (region.shape == self.brightness_dict["W"].shape):
        #     print("dimensions / shapes equal correct")
        # else:
        #     print("dimesnions in subpixel not correct")
        #     return
        # iterate through the dict
        average_bright = np.mean(region)
        # print("avg bright:", average_bright)
        # if average_bright > 200:
        #     return " "
        for char in self.brightness_dict_flat:
            # print("checking new char:", char)
            char_arr = self.brightness_dict_flat[char]
            # print(char_arr)

            # difference = 0
            # # similarity = np.sum((255 - np.abs(region - char_arr)))
            # similarity = 0

            # sim_arr = np.empty(shape=len(region))
            # for y in range(len(region)):

            #     np.append(sim_arr, np.empty(shape=len(region[y])))
            #     for x in range(len(region[y])):
            #         # color_avg_char = char_arr[y][x][0]
            #         color_avg_char_arr = char_arr[y][x]
            #         color_avg_region = region[y][x]
            #         # print(char, color_avg_char_arr, color_avg_region)
            #         this_pixel_diff = abs(region - color_avg_char_arr)
            #         np.append(sim_arr[y], this_pixel_diff)
                    # similarity = similarity + (255 - this_pixel_diff) ** 2
            #         # vanilla
            #         # difference = difference + this_pixel_diff
                    
            #         # squared
            #         difference = difference + (this_pixel_diff * this_pixel_diff)
            # if similarity > highest_similarity:
            #     highest_similarity = similarity
            #     char_holder = char
            ssim_score = metrics.structural_similarity(region, char_arr, full=True)
            # print(ssim_score[0])
            score = round(ssim_score[0], 3)
            # print(char, "rounded ssim", score)
            if score > best_ssim:
                best_ssim = score
                char_holder = char
            # print("\n\n")
            # print(f"SSIM Score: ", round(ssim_score[0], 2))
            # print(region)
            # print(char_arr)
            # # print("simarr\n", sim_arr)
            
            # print("similarity:", similarity)
            # print("\n\n")
            # if difference < lowest_difference:
            #     lowest_difference = difference
            #     char_holder = char
        return char_holder

        

long = Ramp(long_ramp)
long.cache_ramp_maps()
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
d = ImageDraw.Draw(ascii_image)
# ascii_image = Image.open(image_path).convert("RGBA")

# print(img_to_convert)

print("should be height:", len(img_to_convert))
print("should be width:", len(img_to_convert[0]))

total = 0
final_amt = vertical_resolution * horizontal_resolution
for y in range(vertical_resolution):
    for x in range(horizontal_resolution):
        # if total > 2:
            # break
        print(f"{total} / {final_amt}")
        # print(y, x)
        current_y = y * char_target_height
        current_x = x * char_target_width
        subregion = img_to_convert[current_y:current_y+char_target_height, current_x:current_x+char_target_width]
        # print(subregion)
        # char_for_img = long.get_char_region_subpixel(subregion)
        char_for_img = long.get_char_region_basic(subregion)
        pastable_image = Image.fromarray(np.uint8(subregion)).convert('RGBA')
        pastable_image = Image.new('RGBA', (char_target_width, char_target_height))
        # d = ImageDraw.Draw(pastable_image)
        d.text((current_x, current_y), char_for_img, fill="black", anchor="lt", font=font)

        # pastable_image.show()
        # ascii_image.paste(pastable_image, (current_x, current_y), pastable_image)

        # pastable_image.show(title=f"y{y}_x{x}")
        # cv2.imshow(f"subregion y:{current_y} x{current_x}", subregion)
        # cv2.waitKey(0)
        total = total + 1

ascii_image.show()
# ascii_image.save('./abcdef2.png')
print(f"{horizontal_resolution} x {vertical_resolution} with a total of {final_amt} regions")
print("--- Image Took %s seconds ---" % (time.time() - start_time))
# print(long.ramp_string)
