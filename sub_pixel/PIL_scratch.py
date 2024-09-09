from PIL import Image, ImageDraw, ImageFont

font_size = 100
font = ImageFont.truetype("../fonts/Courier_New_Bold.ttf", size=font_size)
im = Image.new("RGB", (int(font_size / 1.6) * 6, int(font_size / 1.6)), "white")
d = ImageDraw.Draw(im)

string = "ABCDEF"
for i in range(len(string)):
    d.text((i * (font_size / 1.6), 0), string[i], fill="black", anchor="lt", font=font)

# d.text((0, 0), "ABCDEF", fill="black", anchor="lt", font=font)
im.show()
