from exif import Image
from datetime import datetime

def get_time(image):
    # potrebujeme otevrit obrazek
    # zmenit ho na objekt Image (cast exif knihovny)
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        # potrebujeme datatime_orginal
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time
    
print(get_time("astropi-iss-speed-en-resources\photo_0676.jpg"))

def get_time_delta(image1, image2):
    # potrebujeme znat rozdil v case vyfoceni dvou obrazku
    pass

# dal potrebujeme najit stejne body na 2 obrazcich
# potom muzeme spocitat rozdil v pozici tehle bodu na tech 2 obrazcich
# z toho by nejak mohlo jit spocitat rychlost (snad)