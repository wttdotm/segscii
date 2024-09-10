import os
import math
import numpy as np
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
from skimage import metrics

# Start timing the execution
start_time = time.time()

# ASCII character ramps
empty = " "
full = "X"
long_ramp = "AMQW#BNHERmKdAGbX8SDOPUkwZF69heT0a&xV%Cs4fY52Lonz3ucJvItr}{li?1][7<>=)(+*|!/-,_~.'` "
short_ramp = "ME$sj1|-^` "
punc_ramp = "@%#*+=-:. "
ramp_437 = '▓▒░ '

# Font and image settings
font_path = "../fonts/Courier_New_Bold.ttf"
font_w_h_aspect_ratio = 1  # True for Courier, based on tests
font_size_to_pixel_ratio = 8/5  # Also true for Courier based on tests

# Input image and output resolution
test_tree = "./test_tree.png"
mandelbrot = "./mandelbrot.jpg"
spiral = "./spiral.jpg"
pure_white = "./pure_white.png"
houndstooth = "./houndstooth.jpg"
ABCDEF2 = "./abcdef2.png"
ABCDEF = "./abcdef.png"

horizontal_resolution = 6  # Can be changed to whatever. Bigger is probably better.

# Load the image in grayscale
img_to_convert = cv2.imread(ABCDEF, cv2.IMREAD_GRAYSCALE)
cv2.imshow("original im", img_to_convert)
cv2.waitKey(0)

# Get image dimensions
i_width = img_to_convert.shape[1]
i_height = img_to_convert.shape[0]
print(f"Image dimensions: {i_width} x {i_height}")
print(f"Horizontal resolution: {horizontal_resolution}")

# Calculate character dimensions
char_target_width = int(i_width / horizontal_resolution)
char_target_height = int(char_target_width * font_w_h_aspect_ratio)
print(f"Character dimensions: {char_target_width} x {char_target_height}")

# Calculate font size
char_display_width = char_target_width * font_size_to_pixel_ratio
char_display_height = char_display_width * font_w_h_aspect_ratio
print(f"Font dimensions: {char_display_width} x {char_display_height}")

# Calculate vertical resolution
vertical_resolution = int(i_height / char_target_height)
print(f"ASCII dimensions: {horizontal_resolution} x {vertical_resolution}")

# Create font
font = ImageFont.truetype(font_path, size=int(char_display_width))

class Ramp:
    def __init__(self, string):
        self.ramp_string = string
        self.brightness_dict = {}
        self.brightness_dict_flat = {}
        self.char_dimensions = {"x": 0, "y": 0}

    def get_char_basic(self, num):
        return self.ramp_string[math.floor((num / 255) * len(self.ramp_string)) - 1]

    def get_char_region_basic(self, region):
        average_bright = np.mean(region)
        # if average_bright > 200:
        #     return " "
        new_array = region.flatten()
        average_brightness = np.average(new_array)
        char = self.get_char_basic(average_brightness)
        return char

    def cache_ramp_maps(self):
        for char in self.ramp_string:
            char_map_pil = Image.new('L', (char_target_width, char_target_height), color="white")
            d = ImageDraw.Draw(char_map_pil)
            d.text((0, 0), char, fill="black", anchor="lt", font=font)
            char_map_cv2 = np.array(char_map_pil)
            self.brightness_dict_flat[char] = char_map_cv2

    def get_char_region_subpixel(self, region, method="ssim"):
        char_holder = ''

        if method == "ssim":
            best_ssim = 0
            for char in self.brightness_dict_flat:
                char_arr = self.brightness_dict_flat[char]
                ssim_score = metrics.structural_similarity(region, char_arr, full=True)
                score = round(ssim_score[0], 3)
                if score > best_ssim:
                    best_ssim = score
                    char_holder = char
            return char_holder
            
        elif method == "similarity_manual":
            highest_similarity = 0
            for char in self.brightness_dict_flat:
                similarity = 0
                char_arr = self.brightness_dict_flat[char]
                for y in range(len(region)):
                    for x in range(len(region[y])):
                        this_pixel_diff = abs(region[y][x] - char_arr[y][x])
                        similarity += ((255 - this_pixel_diff) ** 2)
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        char_holder = char
            return char_holder
        
        elif method == "similarity_np":
            highest_similarity = 0
            highest_similarity_arr = []
            for char in self.brightness_dict_flat:
                # print(char)
                char_arr = self.brightness_dict_flat[char] 
                similarity = np.sum((255 - np.abs(region - char_arr)) ** 2)
                # print(similarity)
                if similarity > highest_similarity:
                        # print(highest_similarity_arr)
                        highest_similarity = similarity
                        highest_similarity_arr = np.abs(region - char_arr)
                        # print(highest_similarity_arr)
                        char_holder = char
            if char_holder != " ":
                print(f"chosen char: '{char_holder}")
                print("char arr")
                print(self.brightness_dict_flat[char_holder])
                print("region")
                print(region)
                print("highest similarity arr")
                print(highest_similarity_arr)
                print("\n\n\n")
            return char_holder
        
        elif method == "difference_np":
            # average_bright = np.mean(region)
            lowest_difference_arr = []
            # if average_bright > 200:
            #     return " "
            lowest_difference = float("inf")
            for char in self.brightness_dict_flat:
                char_arr = self.brightness_dict_flat[char]
                difference = np.sum(np.abs(region - char_arr) ** 2)
                # print(char, difference)
                if difference < lowest_difference:
                    lowest_difference = difference
                    lowest_difference_arr = np.abs(region - char_arr)
                    char_holder = char
            if char_holder != " ":
                print(f"chosen char: '{char_holder}")
                print("char arr")
                print(self.brightness_dict_flat[char_holder])
                print("region")
                print(region)
                print("lowest difference arr")
                print(lowest_difference_arr)
                print("\n\n\n")
            return char_holder
                
# Create and cache the ramp
long = Ramp(long_ramp)
long.cache_ramp_maps()

# Create the output image
ascii_image = Image.new("RGB", (int(char_target_width * horizontal_resolution), int(char_target_height * vertical_resolution)), "white")
d = ImageDraw.Draw(ascii_image)

print(f"Image dimensions: {len(img_to_convert[0])} x {len(img_to_convert)}")

total = 0
final_amt = vertical_resolution * horizontal_resolution

# Generate ASCII art
for y in range(vertical_resolution):
    for x in range(horizontal_resolution):
        print(f"Processing: {total} / {final_amt}")
        current_y = y * char_target_height
        current_x = x * char_target_width
        subregion = img_to_convert[current_y:current_y+char_target_height, current_x:current_x+char_target_width]
        
        # Choose between basic or subpixel method
        # char_for_img = long.get_char_region_subpixel(subregion, "ssim")
        # char_for_img = long.get_char_region_subpixel(subregion, "similarity_manual")
        # char_for_img = long.get_char_region_subpixel(subregion, "similarity_np")
        char_for_img = long.get_char_region_subpixel(subregion, "difference_np")
        # char_for_img = long.get_char_region_basic(subregion)
        
        d.text((current_x, current_y), char_for_img, fill="black", anchor="lt", font=font)
        total += 1

# Display the result
ascii_image.show()

print(f"ASCII dimensions: {horizontal_resolution} x {vertical_resolution}")
print(f"Total regions processed: {final_amt}")
print(f"Execution time: {time.time() - start_time:.2f} seconds")