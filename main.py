import time
from newton_levy import couleur
import cv2
import numpy as np

# Paramètres à ajuster en fonction de la vidéo.
BLACK = 350
BACKGROUND_COLOR = (0, 0, 0)
THRESHOLD = 70000  # pixels


def luminosite(pixel):
    r, g, b = pixel
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b  # ITU-R BT.709
    return lum


def destruction_noir(images_list_path: str, n: int):
    """On reconnait les pixels noirs sur la première image, puis on les retire sur tous les n autres,
    on va aussi réduire l'image à ce qui nous interesse pour le calcul"""
    start_image = images_list_path + "image_1.png"
    start = cv2.imread(start_image)
    alpha = np.sum(start, axis=-1) > BLACK
    array = np.where(alpha == True)
    alpha = np.uint8(alpha * 255)
    x, y = 0, array[0][0]
    _, w, _ = start.shape
    h = array[0][-1]
    alpha = alpha[y:y + h, x:x + w]
    for i in range(n-1):
        path = images_list_path + "image_" + str(i + 1) + ".png"
        im = cv2.imread(path)
        im = im[y:y + h, x:x + w]
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
        print(i, end=": ")
        if is_black(im, initial_black):
            return i
    else:
        return -1


def is_black(im, initial_black, threshold=THRESHOLD):
    """Détermine le nombre de pixels noirs de l'image, si celui est dépassé, on a l'indice du premier noir."""
    n = np.sum(im, axis=-1) < BLACK
    n = np.count_nonzero(n)
    print(n - initial_black)
    return (n - initial_black) > threshold


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


def get_epaisseur(images_list_path: str, output_path: str, first_b: int, color: dict):
    black = cv2.imread(images_list_path + "image" + str(first_b) + ".png")
    h, w, _ = black.shape
    found = dict()


# Combiner read_vid et destruction_noir
def main():
    t = time.time()
    video_path = "videos/5bis.mp4"
    image_path = "processed_images/"
    output_path = "output_images"
    n = read_vid(video_path, 120)
    print(n)
    destruction_noir("processed_images/", n)
    t_elapsed = time.time() - t
    print("Done. Runtime: " + str(t_elapsed) + "s.")
    first = first_black(image_path, n)
    color = couleur()
    get_epaisseur(image_path, output_path, first, color)




if __name__ == "__main__":
    main()
