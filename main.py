import imageio.v3 as im3
from matplotlib.pyplot import imshow, show
from PIL import Image

BLACK = 30
BACKGROUND_COLOR = (255, 255, 255)


# noinspection SpellCheckingInspection
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


def extraire_couleurs(img):
    pass
    # TODO


def main():
    img = im3.imread('images/image.png')
    imshow(img)
    show()
    imshow(nettoyage_image(img))
    show()
    # TODO


main()
