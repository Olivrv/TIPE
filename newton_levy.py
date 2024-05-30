import numpy as np
import scipy.integrate as integrate

VISIBLE = ((3.8 * (10 ** (-7))), (6.5 * (10 ** (-7))))
BLEU = ((3.8 * (10 ** (-7))), (4.90 * (10 ** (-7))))
VERT = ((4.90 * (10 ** (-7))), (5.7 * (10 ** (-7))))
ROUGE = ((5.7 * (10 ** (-7))), (6.5 * (10 ** (-7))))


def eclairement(dif_marche, l_onde):
    """On relie l'éclairement à la difference de marche et à la longueur d'onde (ie le nombre d'onde)."""
    return 1 - np.cos((2 * np.pi * dif_marche) / l_onde)


def eclairement_visible(dif_marche):
    """On l'intègre sur le visible pour avoir l'éclairement total."""
    return integrate.quad(eclairement, VISIBLE[0], VISIBLE[1], args=dif_marche)[0]


def eclairement_bleu(dif_marche):
    """On intègre désormais sur la longueur d'onde correspondant au bleu."""
    return integrate.quad(eclairement, BLEU[0], BLEU[1], args=dif_marche)[0]


def eclairement_vert(dif_marche):
    """On intègre désormais sur la longueur d'onde correspondant au bleu."""
    return integrate.quad(eclairement, VERT[0], VERT[1], args=dif_marche)[0]


def eclairement_rouge(dif_marche):
    """On intègre désormais sur la longueur d'onde correspondant au bleu."""
    return integrate.quad(eclairement, ROUGE[0], ROUGE[1], args=dif_marche)[0]
