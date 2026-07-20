import os
import csv
import gc
import cv2
import numpy as np
import tensorflow as tf
import keras_hub
from scipy.special import iv


class DINO_Bessel_FeatureExtractor:

    def __init__(
        self,
        image_size=(518, 518),
        preset="dinov2_with_registers_small",
        input_dir=r"..\Output\training\05_background_removal",
        output_dir=r"..\Output\training\06_csv_data",
    ):

        self.image_size = image_size
        self.input_dir = input_dir
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        print("Loading DINOv2 model...")

        self.backbone = keras_hub.models.DINOV2Backbone.from_preset(
            preset
        )

        print("Model Loaded Successfully.")

    ###########################################################################

    def preprocess(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)

        image = image.astype(np.float32) / 255.0

        image = np.expand_dims(image, axis=0)

        return image

    ###########################################################################

    def extract_feature(self, image):

        output = self.backbone(image, training=False)

        if isinstance(output, dict):

            if "sequence_output" in output:
                feature = output["sequence_output"]

            elif "encoded_tokens" in output:
                feature = output["encoded_tokens"]

            else:
                feature = list(output.values())[0]

        else:

            feature = output

        feature = tf.convert_to_tensor(feature)

        if len(feature.shape) == 4:

            feature = tf.reduce_mean(feature, axis=[1, 2])

        elif len(feature.shape) == 3:

            feature = tf.reduce_mean(feature, axis=1)

        elif len(feature.shape) == 2:

            pass

        else:

            raise ValueError(
                f"Unexpected feature shape: {feature.shape}"
            )

        feature = feature.numpy().astype(np.float32).flatten()

        return feature

    ###########################################################################

    def bessel_embedding(self, feature):

        feature = np.asarray(feature, dtype=np.float32)

        bessel = iv(0, feature).astype(np.float32)

        embedding = np.concatenate(
            [feature, bessel],
            axis=0
        )

        return embedding

    ###########################################################################

    def process_image(self, image_path):

        image = self.preprocess(image_path)

        feature = self.extract_feature(image)

        embedding = self.bessel_embedding(feature)

        del image
        del feature

        return embedding

    ###########################################################################

    def process_folder(self):

        extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        )

        output_csv = os.path.join(
            self.output_dir,
            "feature_extraction.csv",
        )

        header_written = False

        total_images = 0
        skipped = 0

        with open(output_csv, "w", newline="") as csv_file:

            writer = csv.writer(csv_file)

            for root, _, files in os.walk(self.input_dir):

                class_name = os.path.basename(root)

                for file in sorted(files):

                    if not file.lower().endswith(extensions):
                        continue

                    image_path = os.path.join(root, file)

                    try:

                        embedding = self.process_image(image_path)

                        if not header_written:

                            header = (
                                ["Image", "Class"]
                                + [
                                    f"Feature_{i+1}"
                                    for i in range(len(embedding))
                                ]
                            )

                            writer.writerow(header)

                            header_written = True

                        row = [file, class_name]

                        row.extend(embedding.tolist())

                        writer.writerow(row)

                        total_images += 1

                        print(
                            f"[{total_images}] "
                            f"{class_name} -> {file}"
                        )

                        del embedding
                        del row

                        if total_images % 50 == 0:
                            gc.collect()

                    except Exception as e:

                        skipped += 1

                        print(f"Skipped: {image_path}")

                        print(e)

                        gc.collect()

        print("\n===================================")
        print("Feature Extraction Completed")
        print("===================================")
        print("Processed :", total_images)
        print("Skipped   :", skipped)
        print("CSV Saved :", output_csv)
        print("===================================")


###############################################################################

# if __name__ == "__main__":
#
#     extractor = DINO_Bessel_FeatureExtractor()
#
#     extractor.process_folder()