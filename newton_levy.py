import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

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
    return integrate.quad(lambda l: eclairement(dif_marche, l), BLEU[0], BLEU[1], )[0]


def eclairement_vert(dif_marche):
    """On intègre désormais sur la longueur d'onde correspondant au bleu."""
    return integrate.quad(lambda l: eclairement(dif_marche, l), VERT[0], VERT[1])[0]


def eclairement_rouge(dif_marche):
    """On intègre désormais sur la longueur d'onde correspondant au bleu."""
    return integrate.quad(lambda l: eclairement(dif_marche, l), ROUGE[0], ROUGE[1])[0]


def rgb(dif_marche):
    total = eclairement_visible(dif_marche)
    (r, g, b) = (eclairement_rouge(dif_marche)/total,
                 eclairement_vert(dif_marche)/total,
                 eclairement_bleu(dif_marche)/total)
    return r, g, b


def plot(epaisseur_max, N=300):
    couleurs = []
    for i in range(1, N):
        dif_marche = i * epaisseur_max/N
        couleurs.append(rgb(dif_marche))
        plt.plot(range(1, N), [dif_marche for i in range(1, N)], color=rgb(dif_marche))

    plt.title("Echelle des teintes de Newton")
    plt.ylabel("Différence de marche")
    plt.show()

plot(0.00001)