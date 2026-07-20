import cv2
import numpy as np
from scipy.special import pbdv


class ParabolicFilter:

    def __init__(self, order=2, kernel_size=11, scale=0.5):
        self.order = order
        self.kernel_size = kernel_size
        self.scale = scale

    def create_kernel(self):
        """
        Create Parabolic Cylinder Filter Kernel
        """

        ax = np.linspace(
            -self.kernel_size // 2,
            self.kernel_size // 2,
            self.kernel_size
        ) * self.scale

        xx, yy = np.meshgrid(ax, ax)

        zz = np.sqrt(xx ** 2 + yy ** 2)

        pc_values, _ = pbdv(self.order, zz)

        kernel = pc_values / (np.sum(np.abs(pc_values)) + 1e-6)

        return kernel

    def apply(self, image):
        kernel = self.create_kernel()
        filtered = cv2.filter2D(image, -1, kernel)
        return filtered

    def process(self, input_path, output_path):
        image = cv2.imread(input_path)
        if image is None:
            raise FileNotFoundError(f"Image not found : {input_path}")
        filtered = self.apply(image)
        cv2.imwrite(output_path, filtered)
        return filtered