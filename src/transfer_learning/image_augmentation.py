"""
This class provides augmentation for the training dataset.
The augmentations applied to the dataset will be:
Left-Right Flip (horizontal flip),
Up-Down Flip (vertical flip),
Random rotation within 45 degree range,
Random Shear within 20 degree range,
Rarely, the augmentation will also add Gaussian noise to the image.
Version: 08/08/2020
"""
import numpy as np
from imgaug import augmenters


class Augmentation:
    def __init__(self):
        self.augmentations = augmenters.Sequential(
            [
                augmenters.Sometimes(0.5, augmenters.Flipud(0.5)),
                augmenters.Sometimes(0.5, augmenters.Fliplr(0.5)),
                augmenters.Sometimes(0.5, augmenters.Rotate((-45, 45))),
                augmenters.Sometimes(0.25, augmenters.ShearX((-20, 20))),
                augmenters.Sometimes(
                    0.1, augmenters.AdditiveGaussianNoise(scale=(0, 0.1 * 255))
                ),
            ],
            random_order=True,
        )

    def __call__(self, image):
        image = np.array(image)
        return self.augmentations.augment_image(image)
