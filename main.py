import time
import imageio.v3 as im3
from matplotlib.pyplot import imshow, show
from PIL import Image
import cv2
import numpy as np

BLACK = 400
BACKGROUND_COLOR = (0, 0, 0)
THRESHOLD = 100000  # pixels


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
    creer_image_vide_de_taille(img)
    empty = im3.imread("images/empty.png")
    for i in range(h):
        for j in range(l):
            if luminosite(img[i][j]) > BLACK:
                empty[i][j] = img[i][j]

            else:
                empty[i][j] = BACKGROUND_COLOR
    return empty


def destruction_noir(image_list_path: str, n: int):
    """On reconnait les pixels noirs sur la première image, puis on les retire sur tous les n autres"""
    start_image = image_list_path + "image_1.png"
    start = cv2.imread(start_image)
    alpha = np.sum(start, axis=-1) > BLACK
    alpha = np.uint8(alpha * 255)
    for i in range(n-1):
        path = image_list_path + "image_" + str(i + 1) + ".png"
        im = cv2.imread(path)
        modified = np.dstack((im, alpha))
        cv2.imwrite(path, modified)
    return True


def first_black(images_list_path, n) -> int:
    """Dans une liste d'image située en image_list_path, de forme image_n, avec n le numéro de l'image,
     renvoie le premier indice où apparait une lame sombre"""
    initial_black = np.count_nonzero(np.sum(cv2.imread(images_list_path + "image_1.png"), axis=-1) < BLACK)
    for i in range(1, n-1):
        image_path = images_list_path + "image_" + str(i) + ".png"
        im = cv2.imread(image_path)
        if is_black(im, initial_black):
            return i
    else:
        return -1


def is_black(im, initial_black,threshold=THRESHOLD):
    """Approche : déterminer un pourcentage de noir de l'image, si celui est dépassé, on a le noir ?"""
    n = np.sum(im, axis=-1) < BLACK
    n = np.count_nonzero(n)
    print(n - initial_black)
    return n - initial_black > threshold


def read_vid(path, step=1):
    vid = cv2.VideoCapture(path)
    count = 1
    step_count = 0
    while vid.isOpened():
        success, image = vid.read()
        if success:
            cv2.imwrite('processed_images/image_%d.png' % count, image)
            count += 1
            step_count += step  # i.e. at 'step' fps, this advances one second
            vid.set(cv2.CAP_PROP_POS_FRAMES, step_count)
        else:
            vid.release()
            break
    return count


# Combiner read_vid et destruction_noir
def main():
    t = time.time()
    video_path = "videos/5bis.mp4"
    image_path = "processed_images/"
    n = read_vid(video_path, 120)
    print(n)
    destruction_noir("processed_images/", n)
    t_elapsed = time.time() - t
    print("Done. Runtime: " + str(t_elapsed) + "s.")
    print(first_black(image_path, n))


if __name__ == "__main__":
    main()
