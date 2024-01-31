from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


im = Image.open('Zeeman Effect\TEST.png')
im = im.convert('LA')
x, y = im.size
# im.show()
xPixels = np.arange(0, x)

px = im.load()
intensity = np.zeros(x)
for i in range(x):
    intensity[i] = px[i,0][0]

plt.plot(xPixels, intensity)
plt.show()