import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt


def Eclairement(sigma, d):
    return 1 - np.cos(2 * np.pi * d * sigma * (10 ** (-9)))


def Eclairementbis(d):
    return si.quad(Eclairement, 1 / (4 * (10 ** (-7))), 1 / (7.5 * (10 ** (-7))), args=(d,))


def A1(d):
    return si.quad(Eclairement, 2.07 * (10 ** 6), 2.31 * (10 ** 6), args=(d,))


def A2(d):
    return si.quad(Eclairement, 1.71 * (10 ** 6), 1.95 * (10 ** 6), args=(d,))


def A3(d):
    return si.quad(Eclairement, 1.3 * (10 ** 6), 1.54 * (10 ** 6), args=(d,))


def B1(d):
    return A1(d)[0] / (A1(d)[0] + A2(d)[0] + A3(d)[0])


def B2(d):
    return A2(d)[0] / (A1(d)[0] + A2(d)[0] + A3(d)[0])


def B3(d):
    return A3(d)[0] / (A1(d)[0] + A2(d)[0] + A3(d)[0])


a, b = 1000, 300
D1, D2, D3 = [], [], []
for i in range(1, 2 * b + 2):
    D1.append(Eclairementbis(-a * (-1.00001 + (i - 1) / b))[0] \
              / 2 / ((4 * (10 ** (-7))) - 1 / (7.5 * (10 ** (-7)))) * B1(-a * (-1.00001 + (i - 1) / b)))
    D2.append(Eclairementbis(-a * (-1.00001 + (i - 1) / b))[0] \
              / 2 / ((4 * (10 ** (-7))) - 1 / (7.5 * (10 ** (-7)))) * B2(-a * (-1.00001 + (i - 1) / b)))
    D3.append(Eclairementbis(-a * (-1.00001 + (i - 1) / b))[0] \
              / 2 / ((4 * (10 ** (-7))) - 1 / (7.5 * (10 ** (-7)))) * B3(-a * (-1.00001 + (i - 1) / b)))
    plt.plot([(-a)*(-1+(i-1))/b, 0], [(-a)*(-1+(i-1))/b, 1], color=(D3[i-1], D2[i-1], D3[i-1]))
    plt.title("Echelle des teintes de Newton")
    plt.ylabel("Différence de marche (nm)")
plt.show()
