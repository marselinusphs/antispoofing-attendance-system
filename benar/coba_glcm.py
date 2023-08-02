import numpy as np
from skimage.feature import graycomatrix, graycoprops

image = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 2, 2, 2],
                  [2, 2, 3, 3]], dtype=np.uint8)

result = graycomatrix(image, [1], [0, 45, 90, 135], levels=4, symmetric=False, normed=False)
print(result[:, :, 0, 0], end=" = result[:, :, 0, 0]\n\n")
print(result[:, :, 0, 1], end=" = result[:, :, 0, 1]\n\n")
print(result[:, :, 0, 2], end=" = result[:, :, 0, 2]\n\n")
print(result[:, :, 0, 3], end=" = result[:, :, 0, 3]\n\n")

