import time
from newton_levy import couleur
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paramètres à ajuster en fonction de la vidéo.
BLACK = 350
BACKGROUND_COLOR = (0, 0, 0)
THRESHOLD = 5000  # pixels
EPAISSEUR_MAX = 0.00007  # mètres


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


def distance_euclide(a1, a2):
    x, y, z = a1
    x2, y2, z2 = a2
    return np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)


def closest_color(c, colors):
    """Plus proche couleur dans colors de c dans l'espace en 3 dimensions des couleurs."""
    r, g, b = c
    l = [(color, distance_euclide((r, g, b), color)) for color in colors]
    d = min(l, key=lambda x: x[1])
    return d[0]


def get_epaisseur(images_list_path: str, output_path: str, first_b: int, color: dict, distance_couleur=40):
    path = images_list_path + "image_" + str(first_b) + ".png"
    black = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w, d = black.shape
    colors = set(i[0] for i in color.keys())


    counter = dict()
    epaisseur = np.array([[0.0000001 for _ in range(w)] for _ in range(h)])
    c = 0
    seen_rangee = set()
    # Attention : i la hauteur, j l'abscisse
    for i in range(1, h):
        for j in range(0, w):
            r, g, b, a = black[i][j]
            if a == 0:  # Si le pixel est transparent, on l'ignore.
                epaisseur[i][j] = -255
            else:
                t = int(r) + int(g) + int(b)
                r, g, b = r / t, g / t, b / t  # On obtient les coefficients r, g, b du pixel
                r, g, b = closest_color((r, g, b), colors)
                intersect = [k for k in range(i - distance_couleur, i + distance_couleur + 1) if k in seen_rangee]
                if (r, g, b) in counter.keys():
                    if intersect == []:
                        counter[(r, g, b)] += 1
                        seen_rangee.add(i)
                else:
                    counter[(r, g, b)] = 1
                if ((r, g, b), counter[(r, g, b)]) in color:
                    c += 1
                    epaisseur[i][j] = color[((r, g, b), counter[(r, g, b)])]
                else:
                    if counter[(r, g, b)] > 3:  # Alors c'est du blanc sale
                        epaisseur[i][j] = EPAISSEUR_MAX
                        c += 1
                    else:
                        epaisseur[i][j] = 0
    e_max = np.amax(epaisseur)
    print(e_max)
    grayscale = np.divide(epaisseur, e_max/255)
    print(c/(h*w))
    plt.imshow(grayscale, cmap='gray', vmin=-255, vmax=255)
    cv2.imwrite((output_path + "image_" + str(first_b) + ".png"), grayscale)

    plt.show()


# Combiner read_vid et destruction_noir
def main():
    t = time.time()
    video_path = "videos/6bis.mp4"
    image_path = "processed_images/"
    output_path = "output_images/"
    n = read_vid(video_path, 120)
    print(n)
    destruction_noir("processed_images/", n)

    first = first_black(image_path, n)
    print("First", first)
    color = couleur()

    get_epaisseur(image_path, output_path, first, color)

    t_elapsed = time.time() - t
    print("Done. Runtime: " + str(t_elapsed) + "s.")


if __name__ == "__main__":
    main()
