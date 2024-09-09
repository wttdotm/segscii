from 

def drawFrame (file):
    print(last_4chars(file))
    print(f'./bad_apple/{last_4chars(file)}.png')
    subprocess.run(['mkdir', f'./bad_apple/{last_4chars(file)}'])
    # subprocess.run(['mkdir', f'./single_ladies/{last_4chars(file)}'])
    # get image
    print("hiiiiiiiiiiiiii", file)
    bad_apple = Image.open(f'./bad_apple_frames/{last_4chars(file)}.png')
    width = bad_apple.width
    height = bad_apple.height
    print(width, height)

    # find how big squares will be
    # the finder limit is how many characters we can reasonably fit across in the finder
    print(width / finder_limit)
    square_size = width / finder_limit

    # get a pixel array of it all
    pixels = bad_apple.getdata()

    # now we use the square size to figure out the pixels we care about
    # splitArr = np.array_split(pixels, height)
    # print(len(splitArr))

    # #this will make the pixels easier to work with
    betterPixelArr = []


    # figure out 
    row_num = 0
    for y in range(height):
        if math.floor(y % (0.8 *square_size)) == 0:
            # print(len(betterPixelArr))
            betterPixelArr.append([])
            x = 0
            while x < width:
                # print(int((y * width) + x))
                betterPixelArr[row_num].append(pixels[int((y * width) + x)])
                # BIG SHIFT HAPPENING HEREx += square_size * 
                x += (square_size * 0.6)
            # print(betterPixelArr[row_num])
            row_num += 1



    # for row in betterPixelArr:
    #     # print("_")
    #     result = ""
    #     for pixel in row:
    #         if pixel[0] > 200:
    #             result += "O"
    #         else:
    #             result += " "
    #     print(f"{result}_")

    index = 0
    for row in betterPixelArr:
        # print("_")
        beginning = f"./bad_apple/{last_4chars(file)}/" + clean_numbers_in_string( "{:05d}".format(index))
        # beginning = f"./single_ladies/{last_4chars(file)}/" + clean_numbers_in_string( "{:05d}".format(index))
        result = "_"
        for pixel in row:
            if pixel[0] > 125:
                result += full
            # elif pixel[0] > 175:
            #     result += threeq
            # elif pixel[0] > 125:
            #     result += half
            # elif pixel[0] > 50:
            #     result += oneq
            else:
                result += empty
        cleanFileName = beginning + result + ".wttdotm"
        # print(cleanFileName)
        f = open(cleanFileName, "w")
        f.close()
        index += 1
