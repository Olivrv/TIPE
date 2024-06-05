import cv2
import numpy as np
from main import *
import matplotlib.pyplot as plt

# Première approche, retrait systématique des alpha < 160.


impath = "images/image_357.jpg"
imstart = "images/image_start.png"
imend = "images/image_end.png"
"""
im = cv2.imread(impath)
# Convert to HSV and take V channel
V = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2]

# Threshold V channel at 100 to make alpha channel (A)
_, A = cv2.threshold(V, 150, 255, cv2.THRESH_BINARY)

# Stack A channel onto RGB channels
result = np.dstack((im, A))

# Save result
cv2.imwrite('images/result1.png', result)


Deuxième approche : identifier tous les pixels noirs de la première images (et des n premières), 
les remplacer par le transparent sur celles cis et sur les suivantes
# cv2.imwrite('images/result2.png', nettoyage_image(im))

# Trop lent, et ne remplace pas effectivement par des pixels transparents.

Troisième méthode, mélange des deux, on altère l'alpha chanel selon la luminosité"""

start = cv2.imread(imstart)
end = cv2.imread(imend)

print(end)
alpha = np.sum(start, axis=-1) > 300
n = np.count_nonzero(alpha)
alpha = np.uint8(alpha * 255)
plt.imshow(alpha, interpolation='nearest')

st = np.dstack((start, alpha))
en = np.dstack((end, alpha))
# Save result
cv2.imwrite('images/resultStart.png', st)
cv2.imwrite('images/resultEnd.png', en)

s2 = cv2.imread('images/resultStart.png', cv2.IMREAD_GRAYSCALE)
end_alpha = np.sum(end, axis=-1) > 300
print(end_alpha)
h, w = end_alpha.shape
array = np.where(end_alpha == True)
f1, f2 = array[0][0], array[1][0]
e1, e2 = array[0][-1], array[1][-1]

plt.imshow(end_alpha, interpolation='nearest')
plt.show()
n2 = np.count_nonzero(alpha)
print(n, n2)
# C'est concluant, on garde cette méthode




