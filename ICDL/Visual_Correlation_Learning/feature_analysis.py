import os
import csv
import gc
import cv2
import numpy as np

from skimage.feature import hog
from scipy.special import lambertw


class HOG_Lambert_FA:

    def __init__(self):

        self.image_size = (512, 512)

        self.orientations = 9
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (2, 2)

        self.INP_DIR = r"..\Output\training\05_background_removal"
        self.OUT_DIR = r"..\Output\training\06_csv_data"

        os.makedirs(self.OUT_DIR, exist_ok=True)

    def preprocess(self, image_path):

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0

        return image

    def extract_hog(self, image):

        feature = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True
        )

        return feature

    def lambert_function(self, feature):

        feature = np.real(lambertw(feature))
        feature = feature.astype(np.float32)

        return feature

    def process_image(self, image_path):

        image = self.preprocess(image_path)
        hog_feature = self.extract_hog(image)
        feature = self.lambert_function(hog_feature)

        del image
        del hog_feature

        return feature

    def process_folder(self):

        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif")

        output_csv = os.path.join(
            self.OUT_DIR,
            "feature_analysis.csv"
        )

        header_written = False
        count = 0

        with open(output_csv, "w", newline="") as csvfile:

            writer = csv.writer(csvfile)

            for root, dirs, files in os.walk(self.INP_DIR):

                for file in sorted(files):

                    if not file.lower().endswith(extensions):
                        continue

                    image_path = os.path.join(root, file)
                    class_name = os.path.basename(root)

                    try:

                        feature = self.process_image(image_path)

                        if not header_written:

                            header = (
                                ["Image", "Class"] +
                                [f"Feature_{i+1}" for i in range(len(feature))]
                            )

                            writer.writerow(header)
                            header_written = True

                        row = [file, class_name]
                        row.extend(feature.tolist())

                        writer.writerow(row)

                        count += 1

                        print(f"Processed: {count} | {class_name} | {file}")

                        del feature

                        if count % 100 == 0:
                            gc.collect()

                    except Exception as e:
                        print(f"Error processing {image_path}")
                        print(e)

        print("\n===================================")
        print("Feature Extraction Completed")
        print("Total Images :", count)
        print("Saved :", output_csv)
        print("===================================")


# if __name__ == "__main__":
#
#     FA = HOG_Lambert_FA()
#     FA.process_folder()