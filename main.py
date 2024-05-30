import imageio.v3 as im3
from matplotlib.pyplot import imshow, show
from PIL import Image
import cv2
from newton_levy import couleur

BLACK = 30
BACKGROUND_COLOR = (255, 255, 255)


def creer_image_vide_de_taille(img):
    empty = Image.new('RGB', (img.shape[1], img.shape[0]), (0, 0, 0, 0))
    empty.save('images/empty.png')
    return None


# noinspection SpellCheckingInspection
def luminosite(pixel):
    r, g, b = pixel
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b  # ITU-R BT.709
    return lum


def nettoyage_image(img):
    """On élimine les contours de l'image qui sont sombres (ie arrière-plan),
    on les remplace par la couleur BACKGROUND_COLOR."""
    h, l, _ = img.shape
    empty = im3.imread("images/empty.png")
    for i in range(h):
        for j in range(l):
            if luminosite(img[i][j]) > BLACK:
                empty[i][j] = img[i][j]
            else:
                empty[i][j] = BACKGROUND_COLOR
    return empty


def destruction_noir(image_list: list, n) -> list:
    """On reconnait les pixels noirs sur les n premières images du film,
    qu'on remplace par du blanc sur toutes les images consécutives (nettoyage du noir
    périphérique, présent avant éclatement du film)"""
    noirs = set()
    for image in range(n):
        img = image_list[image]
        h, l, _ = img.shape
        for i in range(h):
            for j in range(l):
                if is_black(img[i][j]):
                    noirs.add((i, j))
    for image in range(len(image_list)):
        for (i, j) in noirs:
            image_list[image][i][j] = BACKGROUND_COLOR
    return image_list


def first_black(images_list: list) -> int:
    """Dans une liste d'image de type im3, renvoie le premier indice où apparait une lame sombre"""
    li = list(images_list)
    for i in range(len(li)):
        image = li[i]
        h, l, _ = image.shape
        for k in range(h):
            for j in range(l):
                if luminosite(image[k][j]) < BLACK:
                    return i
    return -1


def is_black(pixel):
    """Approche : déterminer un pourcentage de noir de l'image, si celui est dépassé, on a le noir ?"""
    # TODO: définir le noir
    pass


def read_vid(path, step=1):
    vid = cv2.VideoCapture(path)
    count = 1
    success, image = vid.read()
    while success:
        cv2.imwrite("processed_images/image_%d.jpg" % count, image)
        success, image = vid.read()
        for i in range(step - 1):
            success, image = vid.read()
        count += 1
    return count


def main():
    video_path = "videos/1.mp4"
    print(read_vid(video_path, 2))
    # TODO Finir le programme


main()
