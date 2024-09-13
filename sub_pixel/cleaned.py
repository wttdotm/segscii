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
# long_ramp = """ !#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~ ¡¢£¤¥¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſƀƁƂƃƄƅƆƇƈƉƊƋƌƍƎƏƐƑƒƓƔƕƖƗƘƙƚƛƜƝƞƟƠơƢƣƤƥƦƧƨƩƪƫƬƭƮƯưƱƲƳƴƵƶƷƸƹƺƻƼƽƾƿǀǁǂǃǄǅǆǇǈǉǊǋǌǍǎǏǐǑǒǓǔǕǖǗǘǙǚǛǜǝǞǟǠǡǢǣǤǥǦǧǨǩǪǫǬǭǮǯǰǱǲǳǴǵǶǷǸǹǺǻǼǽǾǿȀȁȂȃȄȅȆȇȈȉȊȋȌȍȎȏȐȑȒȓȔȕȖȗȘșȚțȜȝȞȟȠȡȢȣȤȥȦȧȨȩȪȫȬȭȮȯȰȱȲȳȴȵȶȸȹȺȻȼȽȾȿɀɁɂɃɄɅɆɇɈɉɊɋɌɍɎɏɐɑɒɓɔɕɖɗɘəɚɛɜɝɞɟɠɡɢɣɤɥɦɧɨɩɪɫɬɭɮɯɰɱɲɳɴɵɶɷɸɹɺɻɼɽɾɿʀʁʂʃʄʅʆʇʈʉʊʋʌʍʎʏʐʑʒʓʔʕʖʗʘʙʚʛʜʝʞʟʠʡʢʣʤʥʦʧʨʩʪʫʬʭʮʯʰʱʲʳʴʵʶʷʸʹʺʻʼʽʾʿˀˁ˂˃˄˅ˆˇˈˉˊˋˌˍˎˏːˑ˒˓˔˕˖˗˘˙˚˛˜˝˞˟ˠˡˢˣˤˬ˭ˮ˯˰˱˲˳˴˵˶˷ͼͽ;Ϳ΄΅Ά·ΈΉΊΌΎΏΐΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΪΫάέήίΰαβγδεζηθικλμνξοπρςστυφχψωϊϋόύώϏϐϑϒϓϔϕϖϗϘϙϚϛϜϝϞϟϠϡϢϣϤϥϦϧϨϩϪϫϬϭϮϯϰϱϲϳϴϵ϶ϷϸϹϺϻϼϽϾϿЀЁЂЃЄЅІЇЈЉЊЋЌЍЎЏАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяѐёђѓєѕіїјљњћќѝўџѠѡѢѣѤѥѦѧѨѩѪѫѬѭѮѯѰѱѲѳѴѵѶѷѸѹѺҔҕҖҗҘҙҚқҜҝҞҟҠҡҢңҤҥҦҧҨҩҪҫҬҭҮүҰұҲҳҴҵҶҷҸҹҺһҼҽҾҿӀӁӂӃӄӅӆӇӈӉӊӋӌӍӎӏӐӑӒӓӔӕӖӗӘәӚӛӜӝӞӟӠӡӢӣӤӥӦӧӨөӪӫӬӭӮӯӰӱӲӳӴӵӶӷӸӹӺӻӼӽӾӿԀԁԂԃԄԅԆԇԈԉԊԋԌԍԎԏԐԑԒԓԔԕԖԗԘԙԚԛԜԝԞԟԠԡԢԣԤԥԦԧԨԩԪԫԬԭԮԯԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖՙ՚՛՜՝՞՟աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆև։֊฿ᴀᴁᴂᴃᴄᴅᴆᴇᴈᴉᴊᴋᴌᴍᴎᴏᴐᴑᴒᴓᴔᴕᴖᴗᴘᴙᴚᴛᴜᴝᴞᴟᴠᴡᴢᴣᴤᴥᴦᴧᴨᴩᴪᴫᴬᴭᴮᴯᴰᴱᴲᴳᴴᴵᴶᴷᴸᴹᴺᴻᴼᴽᴾᴿᵀᵁᵂᵃᵄᵅᵆᵇᵈᵉᵊᵋᵌᵍᵎᵏᵐᵑᵒᵓᵔᵕᵖᵗᵘᵙᵚᵛᵜᵝᵞᵟᵠᵡᵢᵣᵤᵥᵦᵧᵨᵩᵪᵫᵬᵭᵮᵯᵰᵱᵲᵳᵴᵵᵶᵷᵸᵹᵺᵻᵼᵽᵾᵿᶀᶁᶂᶃᶄᶅᶆᶇᶈᶉᶊᶋᶌᶍᶎᶏᶐᶑᶒᶓᶔᶕᶖᶗᶘᶙᶚᶛᶜᶝᶞᶟᶠᶡᶢᶣᶤᶥᶦᶧᶨᶩᶪᶫᶬᶭᶮᶯᶰᶱᶲᶳᶴᶵᶶᶷᶸᶹᶺḆḇḈḉḊḋḌḍḎḏḐḑḒḓḔḕḖḗḘḙḚḛḜḝḞḟḠḡḢḣḤḥḦḧḨḩḪḫḬḭḮḯḰḱḲḳḴḵḶḷḸḹḺḻḼḽḾḿṀṁṂṃṄṅṆṇṈṉṊṋṌṍṎṏṐṑṒṓṔṕṖṗṘṙṚṛṜṝṞṟṠṡṢṣṤṥṦṧṨṩṪṫṬṭṮṯṰṱṲṳṴṵṶṷṸṹṺṻṼṽṾṿẀẁẂẃẄẅẆẇẈẉẊẋẌẍẎẏẐẑẒẓẔẕẖẗẘẙẚẛẜẝẞẟẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹỺỻỼỽỾỿἀἁἂἃἄἅἆἇἈἉἊἋἌἍἎἏἐἑἒἓἔἕἘἙἚἛἜἝἠἡἢἣἤἥἦἧἨἩἪἫἬἭἮἯἰἱἲἳἴἵἶἷἸἹἺἻἼἽἾἿὀὁὂὃὄὅὈὉὊὋὌὍὐὑὒὓὔὕὖὗὙὛὝὟὠὡὢὣὤὥὦὧὨὩὪὫὬὭὮὯὰάὲέὴήὶίὸόὺύὼώᾀᾁᾂᾃᾄᾅᾆᾇᾈᾉᾊᾋᾌᾍᾎᾏᾐᾑᾒᾓᾔᾕᾖᾗᾘᾙᾚᾛᾜᾝᾞᾟᾠᾡᾢᾣᾤᾥᾦᾧᾨᾩᾪᾫᾬᾭᾮᾯᾰᾱᾲᾳᾴᾶᾷᾸᾹᾺΆᾼ᾽ι᾿῀῁ῂῃῄῆῇῈΈῊΉῌ῍῎῏ῐῑῒΐῖῗῘῙῚΊ῝῞῟ῠῡῢΰῤῥῦῧῨῩῪΎῬ῭΅`ῲῳῴῶῷῸΌῺ‘’‚‛“”„‟†‡•…‰′″‴‹›‼‽‾⁄⁞ⁿₐₑₒₓₔ₠₡₢₣₤₥₦₧₨₩₪₫€₭₮₯₰₱₲₳₴₵₶₷₸₹₺₻₼₽₾₿⃰℅ℓ№™Ω℮⅍ⅎ⅓⅔⅛⅜⅝⅞ↄ←↑→↓↔↕↨∂∆∏∑−∕∙√∞∟∩∫≈≠≡≤≥⌂⌐⌠⌡─│┌┐└┘├┤┬┴┼═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬▀▄█▌▐░▒▓■□▪▫▬▲►▼◄◊○◌●◘◙◦☺☻☼♀♂♠♣♥♦♪♫♯ⱠⱡⱢⱣⱤⱥⱦⱧⱨⱩⱪⱫⱬⱭⱮⱯⱰⱱⱲⱳⱴⱵⱶⱷⱸⱹⱺⱻⱼⱽⱾⱿ⸗ꜗꜘꜙꜚꜛꜜꜝꜞꜟ꜠꜡ꞈ꞉꞊Ꞌꞌﬀﬁﬂﬃﬄﬅﬆﬓﬔﬕﬖﬗיִﬞײַﬠﬡﬢﬣﬤﬥﬦﬧﬨ﬩שׁשׂשּׁשּׂאַאָאּבּגּדּהּוּזּטּיּךּכּלּמּנּסּףּפּצּקּרּשּתּוֹבֿכֿפֿﭏﭐﭑﭒﭓﭔﭕﭖﭗﭘﭙﭚﭛﭜﭝﭞﭟﭠﭡﭢﭣﭤﭥﭦﭧﭨﭩﭪﭫﭬﭭﭮﭯﭰﭱﭲﭳﭴﭵﭶﭷﭸﭹﭺﭻﭼﭽﭾﭿﮀﮁﮂﮃﮄﮅﮆﮇﮈﮉﮊﮋﮌﮍﮎﮏﮐﮑﮒﮓﮔﮕﮖﮗﮘﮙﮚﮛﮜﮝﮞﮟﮠﮡﮢﮣﮤﮥﮦﮧﮨﮩﮪﮫﮬﮭﮮﮯﮰﮱ﮲﮳﮴﮵﮶﮷﮸﮹﮺﮻﮼﮽﮾﮿﯀﯁ﯓﯔﯕﯖﯗﯘﯙﯚﯛﯜﯝﯞﯟﯠﯡﯢﯣﯤﯥﯦﯧﯨﯩﯪﯫﯬﯭﯮﯯﯰﯱﯲﯳﯴﯵﯶﯷﯸﯹﯺﯻﯼﯽﯾﯿﰈﱮﱯﱰﱳﱴﱵﲎﲏﲑﲔﲜﲝﲞﲟﲠﲡﲢﲣﲤﲥﲦﲨﲪﲬﲰﳉﳊﳋﳌﳍﳎﳏﳐﳑﳒﳓﳔﳕﳖﳘﳚﳛﳜﳝﳲﳳﳴﴰﴼﴽ﴾﴿ﶈﷲﷴﷺﷻ﷼︠︡︢︣ﹰﹱﹲﹳﹴﹶﹷﹸﹹﹺﹻﹼﹽﹾﹿﺀﺁﺂﺃﺄﺅﺆﺇﺈﺉﺊﺋﺌﺍﺎﺏﺐﺑﺒﺓﺔﺕﺖﺗﺘﺙﺚﺛﺜﺝﺞﺟﺠﺡﺢﺣﺤﺥﺦﺧﺨﺩﺪﺫﺬﺭﺮﺯﺰﺱﺲﺳﺴﺵﺶﺷﺸﺹﺺﺻﺼﺽﺾﺿﻀﻁﻂﻃﻄﻅﻆﻇﻈﻉﻊﻋﻌﻍﻎﻏﻐﻑﻒﻓﻔﻕﻖﻗﻘﻙﻚﻛﻜﻝﻞﻟﻠﻡﻢﻣﻤﻥﻦﻧﻨﻩﻪﻫﻬﻭﻮﻯﻰﻱﻲﻳﻴﻵﻶﻷﻸﻹﻺﻻﻼ￼"""
long_ramp = "AMQW#¿BNHERmKdAGbX8SDOPUkwZF69heT0a&xV%Cs4fY52Lonz3ucJvItr}{li?1][7<>=)(+*|!/-,_~.'` "
# long_ramp = "AMQW#¿BNHERmKdAGbX8SDOPUkwZF69heT0a&xV%Cs4fY52Lonz3ucJvItr}{li?1][7<>=)(+*|!/-,_~.'` ■□▢▣▤▥▦▧▨▩▬▭▮▯▰▱▲△▴▵▷▸▹►▻▼▽▾▿2◁◂◃◄◅◆◇◈◉◊○◌◍◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯◰◱◲◳◴◵◶◷◸◹◺◿▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞"
# long_ramp = "ABDCDEF"
short_ramp = "ME$sj1|-^` "
punc_ramp = "@%#*+=-:. "
ramp_437 = '▓▒░ '

# Font and image settings
font_path = "../fonts/Courier_New_Bold.ttf"
font_w_h_aspect_ratio = 1  # True for Courier, based on tests
font_size_to_pixel_ratio = 8/5  # Also true for Courier based on tests

# Input image and output resolution
test_tree = "./test_tree.png"
test_tree_2 = "./test_tree2.jpg"
mandelbrot = "./mandelbrot.jpg"
spiral = "./spiral.jpg"
pure_white = "./pure_white.png"
houndstooth = "./houndstooth.jpg"
houndstooth_test = "./test_houndstooth.png"
houndstooth_test_2 = "./test_houndstooth_2.png"
ABCDEF2 = "./abcdef2.png"
ABCDEF = "./abcdef.png"
smash_logo = "./smash_logo.png"

path_to_use = test_tree_2

horizontal_resolution = 50  # Can be changed to whatever. Bigger is probably better.
map_size = 3

# Load the image in grayscale
img_to_convert = cv2.imread(path_to_use, cv2.IMREAD_GRAYSCALE)
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
        self.all_chars = {}
        self.brightness_dict = {}
        self.brightness_dict_flat = {}
        self.brightness_dict_small = {}
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
        all_char_image = Image.new("RGBA", (char_target_width * len(self.ramp_string), char_target_height), color="white")
        all_char_image_small = Image.new("RGBA", (char_target_width * len(self.ramp_string), map_size), color="white")
        p = ImageDraw.Draw(all_char_image)
        ps = ImageDraw.Draw(all_char_image_small)
        total = 0
        for char in self.ramp_string:
            char_map_pil = Image.new('L', (char_target_width, char_target_height), color="white")

            d = ImageDraw.Draw(char_map_pil)
            d.text((0, char_target_height / 2), char, fill="black", anchor="lm", font=font)
            p.text((total * char_target_width, char_target_height / 2), char, fill="black", anchor="lm", font=font)
            char_map_pil_smaller = char_map_pil.resize((map_size, map_size))
            all_char_image_small.paste(char_map_pil_smaller, (total * map_size, 0))
            char_map_cv2 = np.array(char_map_pil)
            char_map_cv2_smaller = np.array(char_map_pil_smaller)

            self.brightness_dict_flat[char] = char_map_cv2
            self.brightness_dict_small[char] = char_map_cv2_smaller
            total += 1
        
        all_char_image.show()
        all_char_image_small.show()
        print("done caching rmps")

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

        elif method == "similarity_squish":
            # lowest similarity is 0
            highest_similarity = 0
            char_holder = ''
            # average = np.average(region)
            # if average > 250:
            #     return " "
            region = cv2.resize(region, (map_size, map_size))
            for char in self.brightness_dict_small:
                similarity = 0
                char_arr = self.brightness_dict_small[char]
                # total_subreg = 0
                for y in range(len(region)):
                    for x in range(len(region[y])):
                        # total_subreg += 1
                        if (char_arr[y][x] < 250 and region[x][y] < 250):
                            this_pixel_diff = abs(region[y][x] - char_arr[y][x])
                            similarity += ((255 - this_pixel_diff) ** 2)
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        char_holder = char
                # print(f"total_subreg: {total_subreg}")
            # image = Image.fromarray(np.uint8(region), mode="L").convert("RGBA")
            # d = ImageDraw.Draw(image)
            # d.text((0, 0), char_holder, fill=(255, 0, 0, 128), anchor="lt", font=font)
            # image.show()
            # cv2.waitKey(0)

            return char_holder
            
        elif method == "similarity_manual":
            # lowest similarity is 0
            highest_similarity = 0
            char_holder = ''
            average = np.average(region)
            # if average > 250:
            #     return " "
            for char in self.brightness_dict_flat:
                similarity = 0
                char_arr = self.brightness_dict_flat[char]
                for y in range(len(region)):
                    for x in range(len(region[y])):
                        if (char_arr[y][x] < 250 and region[x][y] < 250):
                            this_pixel_diff = abs(region[y][x] - char_arr[y][x])
                            similarity += ((255 - this_pixel_diff) ** 2)
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        char_holder = char
            # image = Image.fromarray(np.uint8(region), mode="L").convert("RGBA")
            # d = ImageDraw.Draw(image)
            # d.text((0, 0), char_holder, fill=(255, 0, 0, 128), anchor="lt", font=font)
            # image.show()
            # cv2.waitKey(0)

            return char_holder
        
        elif method == "similarity_np":
            highest_similarity = 0
            highest_similarity_arr = []
            for char in self.brightness_dict_flat:
                # print(char)
                char_arr = self.brightness_dict_flat[char] 
                if char == " ":
                    similarity = np.sum((255 - np.abs(region - char_arr)) ** 2)
                else:
                    mask = char_arr != 255
                    similarity = np.sum((255 - np.abs(region[mask] - char_arr[mask]))**2)
                # print(similarity)
                if similarity > highest_similarity:
                        # print(highest_similarity_arr)
                        highest_similarity = similarity
                        highest_similarity_arr = np.abs(region - char_arr)
                        # print(highest_similarity_arr)
                        char_holder = char
            return char_holder

        elif method == "similarity_new":
            lowest_difference = float('inf')
            highest_similarity = 0
            char_holder = ' '  # Default to space
            region_mean = np.mean(region)
            
            for char, char_arr in self.brightness_dict_flat.items():
                if char == ' ':
                    # For space, check if the region is close to white
                    if region_mean > 240:  # You can adjust this threshold
                        return ' '
                    continue
                
                # For other characters, use masked comparison
                mask = char_arr < 255
                if np.sum(mask) == 0:  # If the character is all white, skip it
                    continue
                
                # Calculate coverage (percentage of non-white pixels)
                coverage = np.mean(char_arr < 255)
                
                # Calculate similarity, focusing on matching dark pixels
                similarity = np.sum((255 - region) * (255 - char_arr)) / (np.sum((255 - char_arr)**2) + 1e-6)
        
                # difference = np.mean((region[mask] - char_arr[mask])**2)
                # similarity = np.mean(255 - (region[mask] - char_arr[mask])**2)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    char_holder = char
                # if difference < lowest_difference:
                #     lowest_difference = difference
                #     char_holder = char
            
            return char_holder

        
        elif method == "difference_np":
            # average_bright = np.mean(region)
            lowest_difference_arr = []
            # if average_bright > 200:
            #     return " "
            lowest_difference = float("inf")
            for char in self.brightness_dict_flat:
                char_arr = self.brightness_dict_flat[char]
                # weight = np.where(char_arr < 127, 2, 0.5)
                # print(weight)
                difference = np.sum(weight *  np.abs(region - char_arr))
                print(f"\n{char}:\n{np.abs(region - char_arr)}")
                # print(difference)
                # print(char, weight, difference)
                if difference < lowest_difference:
                    lowest_difference = difference
                    lowest_difference_arr = np.abs(region - char_arr)
                    char_holder = char
            # if char_holder != " ":
                # print(f"chosen char: '{char_holder}")
                # print("char arr")
                # print(self.brightness_dict_flat[char_holder])
                # print("region")
                # print(region)
                # print("lowest difference arr")
                # print(lowest_difference_arr)
                # print("\n\n\n")
            return char_holder
                
# Create and cache the ramp
long = Ramp(long_ramp)
long.cache_ramp_maps()

# Create the output image
# ascii_image = Image.open(path_to_use)
ascii_image = Image.new("RGBA", (int(char_target_width * horizontal_resolution), int(char_target_height * vertical_resolution)), "white")
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
        char_for_img = long.get_char_region_subpixel(subregion, "similarity_squish")
        # char_for_img = long.get_char_region_subpixel(subregion, "similarity_manual")
        # char_for_img = long.get_char_region_subpixel(subregion, "similarity_np")
        # char_for_img = long.get_char_region_subpixel(subregion, "similarity_new")
        # char_for_img = long.get_char_region_subpixel(subregion, "difference_np")
        # char_for_img = long.get_char_region_basic(subregion)
        # d.rectangle(((current_x, current_y), (current_x + char_target_width, current_y + char_target_height)), outline = "blue")
        d.text((current_x, current_y + (char_target_height / 2)), char_for_img, fill=(0, 0, 0, 255), anchor="lm", font=font)
        total += 1

# Display the result
ascii_image.show()

print(f"ASCII dimensions: {horizontal_resolution} x {vertical_resolution}")
print(f"Total regions processed: {final_amt}")
print(f"Execution time: {time.time() - start_time:.2f} seconds")