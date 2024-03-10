import numpy as np
from PIL import Image

with Image.open('../data/patterns/logged_white_rook.png') as image:
    red_image = np.asarray(image)[:45, :45, 0]
    print(red_image.shape)
    np.save('../data/patterns/logged_white_rook.npy', red_image)

with Image.open('../data/patterns/logged_black_rook.png') as image:
    red_image = np.asarray(image)[:45, :45, 0]
    print(red_image.shape)
    np.save('../data/patterns/logged_black_rook.npy', red_image)
