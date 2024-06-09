import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Longueurs d'onde
VISIBLE = ((3.8 * (10 ** (-7))), (6.5 * (10 ** (-7))))
BLEU = ((3.8 * (10 ** (-7))), (4.90 * (10 ** (-7))))
VERT = ((4.90 * (10 ** (-7))), (5.7 * (10 ** (-7))))
ROUGE = ((5.7 * (10 ** (-7))), (6.5 * (10 ** (-7))))


def eclairement(dif_marche, l_onde):
    """On relie l'éclairement à la difference de marche et à la longueur d'onde (ie le nombre d'onde)."""
    return 1 - np.cos((2 * np.pi * dif_marche) / l_onde)


def eclairement_visible(dif_marche):
    """On l'intègre sur le visible pour avoir l'éclairement total."""
    return integrate.quad(lambda l: eclairement(dif_marche, l), VISIBLE[0], VISIBLE[1])[0]


def eclairement_bleu(dif_marche):
    """On intègre désormais sur la longueur d'onde correspondant au bleu."""
    return integrate.quad(lambda l: eclairement(dif_marche, l), BLEU[0], BLEU[1])[0]


def eclairement_vert(dif_marche):
    """On intègre désormais sur la longueur d'onde correspondant au vert."""
    return integrate.quad(lambda l: eclairement(dif_marche, l), VERT[0], VERT[1])[0]


def eclairement_rouge(dif_marche):
    """On intègre désormais sur la longueur d'onde correspondant au rouge."""
    return integrate.quad(lambda l: eclairement(dif_marche, l), ROUGE[0], ROUGE[1])[0]


def rgb(dif_marche):
    total = eclairement_visible(dif_marche)
    (r, g, b) = (eclairement_rouge(dif_marche) / total,
                 eclairement_vert(dif_marche) / total,
                 eclairement_bleu(dif_marche) / total)
    return r, g, b


def plot(epaisseur_max=0.00001, n=500):
    couleurs = []
    for i in range(1, n):
        dif_marche = i * epaisseur_max / n
        couleurs.append(rgb(dif_marche))
        plt.plot(range(1, n), [dif_marche for _ in range(1, n)], color=rgb(dif_marche))

    plt.title("Echelle des teintes de Newton")
    plt.ylabel("Différence de marche (m)")
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    plt.show()


def couleur(epaisseur=0.00001, n=500, precision=2, delta=0.5 * (10**-6)) -> dict:
    """
    épaisseur : epaisseur maximal du film, ie jusqu'où on calcule
    n : nombre de valeurs entre 0 et épaisseur
    précision : précision de la valeur de r, g, b, important pour que les couches multiples soient prises en compte.
    """
    color = dict()
    vu = dict()
    previous_color = (0, 0, 0)
    for i in range(1, n):
        dif_marche = i * epaisseur / n
        r, g, b = rgb(dif_marche)
        c = (round(r, precision), round(g, precision), round(b, precision))
        if c in vu.keys():
            if previous_color != c and abs(color[c, vu[c]] - dif_marche) > delta:
                vu[c] += 1
        else:
            vu[c] = 1
        color[(c, vu[c])] = dif_marche
    return color


if __name__ == "__main__":
    plot()
    print(couleur())
    print(max([i for _, i in couleur().keys()]))
