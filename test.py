import cv2
import numpy as np
from newton_levy import couleur

impath = "/images/image_1863.jpg"
im = cv2.imread(impath)

# Convert to HSV and take V channel
V = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2]

# Threshold V channel at 100 to make alpha channel (A)
_, A = cv2.threshold(V, 100, 255, cv2.THRESH_BINARY)

# Stack A channel onto RGB channels
result = np.dstack((im, A))

# Save result
cv2.imwrite('result.png', result)
