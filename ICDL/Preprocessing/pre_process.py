import os
import cv2


class ImagePreprocessing:

    def __init__(self, resize_size=(512, 512)):
        self.resize_size = resize_size

    def resize(self, image):
        return cv2.resize(image, self.resize_size)

    def noise_removal(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def contrast_enhancement(self, image):

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def process_folder(self, input_folder):

        resize_folder = "..\\Output\\training\\01_Resize"
        noise_folder = "..\\Output\\training\\02_Noise_Removal"
        contrast_folder = "..\\Output\\training\\03_Contrast_Enhancement"

        for root, dirs, files in os.walk(input_folder):

            relative = os.path.relpath(root, input_folder)

            resize_save = os.path.join(resize_folder, relative)
            noise_save = os.path.join(noise_folder, relative)
            contrast_save = os.path.join(contrast_folder, relative)

            os.makedirs(resize_save, exist_ok=True)
            os.makedirs(noise_save, exist_ok=True)
            os.makedirs(contrast_save, exist_ok=True)

            for file in files:

                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

                    img_path = os.path.join(root, file)

                    image = cv2.imread(img_path)

                    if image is None:
                        continue

                    # Resize
                    resized = self.resize(image)
                    cv2.imwrite(os.path.join(resize_save, file), resized)

                    # Noise Removal
                    noise = self.noise_removal(resized)
                    cv2.imwrite(os.path.join(noise_save, file), noise)

                    # Contrast Enhancement
                    contrast = self.contrast_enhancement(noise)
                    cv2.imwrite(os.path.join(contrast_save, file), contrast)

        print("All Images Processed Successfully")
pp = ImagePreprocessing()

pp.process_folder("..\\dataset\\images")