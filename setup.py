import menpo.io as mio
from conversion import landmarkConverter

def process(image, crop_proportion=0.2, max_diagonal=400):
    path = str(image.path)
    lpath = s.replace('.png','_aam.txt')
    lm = landmarkConverter(lpath)
    image.landmarks = lm
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    return image


