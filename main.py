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
n = 1.4  # Indice de la pelliculle savonneuse


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
    for i in range(n - 1):
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
    for i in range(1, n - 1):
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


def clean(epaisseur, j):
    for i in range(0, 115):
        epaisseur[i][j] = 0
    for i in range(360, len(epaisseur)):
        epaisseur[i][j] = 0
    return epaisseur


def get_epaisseur1(images_list_path: str, output_path: str, first_b: int, color: dict, distance_couleur=1):
    n = 1.4  # Indice de la pelliculle savonneuse
    path = images_list_path + "image_" + str(first_b) + ".png"
    black = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w, d = black.shape
    colors = set(i[0] for i in color.keys())

    counter = dict()
    epaisseur = np.array([[0.0000001 for _ in range(w)] for _ in range(h)])
    c = 0
    seen_rangee = set()
    # Attention : i la hauteur, j l'abscisse
    for k in range(1, h):
        i = h - k
        for j in range(0, w):
            r, g, b, a = black[i][j]
            if a == 0:  # Si le pixel est transparent, on l'ignore.
                epaisseur[i][j] = 0
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
                    epaisseur[i][j] = color[((r, g, b), counter[(r, g, b)])]/2*n
                else:
                    if counter[(r, g, b)] > 3:  # Alors c'est du blanc sale
                        epaisseur[i][j] = epaisseur[i-1][j]
                        c += 1
                    else:
                        epaisseur[i][j] = 0
    e_max = np.amax(epaisseur)
    print(e_max)
    grayscale = np.divide(epaisseur, e_max/255)
    print(c/(h*w))
    # plt.imshow(grayscale, cmap='gray', vmin=-255, vmax=255)
    # clean(epaisseur, int(w/2))
    start, end = 105, 500
    plt.plot(range(start, end), [epaisseur[i][int(w/2)] for i in range(start, end)])
    cv2.imwrite((output_path + "image_" + str(first_b) + ".png"), grayscale)

    plt.show()


def pixel_color(img, i, j, colors):
    r, g, b, a = img[i][j]
    if a == 0:
        return 0, 0, 0
    try:
        t = int(r) + int(g) + int(b)
    except RuntimeWarning:
        print(r, g, b, t)
    if t == 0:
        return 0, 0, 0
    else:
        r, g, b = r / t, g / t, b / t
        return closest_color((r, g, b), colors)


def get_epaisseur(images_list_path: str, output_path: str, first_b: int, distance_couleur=40):
    path = images_list_path + "image_" + str(first_b) + ".png"
    black = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w, d = black.shape
    color = couleur(EPAISSEUR_MAX, h)
    print(color)
    # on va tenter d'assigner à chaque pixel une couleur parmi celles dont on dispose dans 'couleurs'
    colors = set(i[0] for i in color.keys())
    # on crée un np array de taille h*w, qu'on définit avec une valeur de l'ordre de celles avec lesquelles on travaille
    epaisseur = np.array([[0.0000001 for _ in range(w)] for _ in range(h)])
    c = 0  # on veut compter le nombre de pixels auquels on a pu assigner une couleur
    counter = dict()
    couleur_vues = np.array([[(0, 0, 0) for _ in range(w)] for _ in range(h)])
    for k in range(1, h):
        i = h - k
        for j in range(w):
            r, g, b, a = black[i][j]
            previous_epaisseur = epaisseur[i - 1][j]
            if a == 0:  # Si le pixel est transparent, on l'ignore.
                epaisseur[i][j] = 0
            else:
                r, g, b = pixel_color(black, i, j, colors)
                couleur_vues[i][j] = (r, g, b)
                if (r, g, b) == (0, 0, 0):
                    epaisseur[i][j] = 0
                else:
                    if (r, g, b) in counter.keys():
                        previous_equal = [k for k in range(max(0, i - distance_couleur), i)
                                          if (couleur_vues[k][j] == (r, g, b)).all()]
                        if ((r, g, b) == couleur_vues[i-1][j]).all():
                            epaisseur[i][j] = previous_epaisseur
                            c += 1
                        else:
                            if previous_equal == []:
                                c += 1
                                if ((r, g, b), counter[(r, g, b)]) in color.keys():
                                    epaisseur[i][j] = color[((r, g, b), counter[(r, g, b)])]
                                    counter[(r, g, b)] += 1
                                else:
                                    # print(((r, g, b), counter[(r, g, b)]))
                                    epaisseur[i][j] = previous_epaisseur
                            else:
                                l = max(previous_equal)
                                c += 1
                                epaisseur[i][j] = epaisseur[l][j]
                    else:
                        counter[(r, g, b)] = 1
                        if ((r, g, b), counter[(r, g, b)]) in color.keys():
                            c += 1
                            epaisseur[i][j] = color[((r, g, b), counter[(r, g, b)])]
                        else:
                            epaisseur[i][j] = 0
    e_max = np.amax(epaisseur)
    print(e_max)
    print(counter)
    grayscale = np.divide(epaisseur, e_max)
    print(c / (h * w))
    # plt.imshow(grayscale, cmap='gray', vmin=0, vmax=e_max)
    start, end = 0, 400
    plt.plot(range(start, end), [epaisseur[end - i][int(w / 2)] for i in range(0, end - start)])
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
    get_epaisseur1(image_path, output_path, first, color)

    t_elapsed = time.time() - t
    print("Done. Runtime: " + str(t_elapsed) + "s.")


if __name__ == "__main__":
    main()
