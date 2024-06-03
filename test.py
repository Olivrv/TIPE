import cv2
import numpy as np
from newton_levy import couleur
from main import nettoyage_image

"""Première approche, retrait systématique des alpha < 160."""


impath = "images/image_357.jpg"

im = cv2.imread(impath)
print(type(im))
# Convert to HSV and take V channel
V = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2]

# Threshold V channel at 100 to make alpha channel (A)
_, A = cv2.threshold(V, 160, 255, cv2.THRESH_BINARY)

# Stack A channel onto RGB channels
result = np.dstack((im, A))

# Save result
cv2.imwrite('images/result1.png', result)


"""Deuxième approche : identifier tous les pixels noirs de la première images (et des n premières), 
les remplacer par le transparent sur celles cis et sur les suivantes"""
cv2.imwrite('images/result2.png', nettoyage_image(im))
