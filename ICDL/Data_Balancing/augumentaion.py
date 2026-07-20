import os
import cv2
import random
import shutil
from tqdm import tqdm
import albumentations as A

class data_balancing:

    def __init__(self):
        pass


    def process_folder(self, INPUT_DIR, OUTPUT_DIR):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.Affine(scale=(0.9, 1.1),
                     translate_percent=(0.05, 0.05),
                     rotate=(-20, 20),
                     shear=(-10, 10),
                     p=0.5)
        ])
        class_counts = {}

        for cls in os.listdir(INPUT_DIR):
            cls_path = os.path.join(INPUT_DIR, cls)

            if os.path.isdir(cls_path):
                images = [
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                class_counts[cls] = len(images)

        print("Image Count:")
        for k, v in class_counts.items():
            print(k, ":", v)

        # Target count
        max_count = max(class_counts.values())
        print("\nTarget Images =", max_count)

        # Balance Dataset
        for cls in class_counts:
            src_folder = os.path.join(INPUT_DIR, cls)
            dst_folder = os.path.join(OUTPUT_DIR, cls)
            os.makedirs(dst_folder, exist_ok=True)
            image_list = [
                f for f in os.listdir(src_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            # Copy original images
            for img_name in image_list:
                shutil.copy(
                    os.path.join(src_folder, img_name),
                    os.path.join(dst_folder, img_name)
                )

            current = len(image_list)
            aug_index = 1
            print(f"\nProcessing {cls}")


            # Random Selection
            while current < max_count:
                img_name = random.choice(image_list)
                img_path = os.path.join(src_folder, img_name)
                image = cv2.imread(img_path)
                augmented = transform(image=image)
                aug_img = augmented["image"]
                save_name = f"aug_{aug_index}.jpg"
                cv2.imwrite(
                    os.path.join(dst_folder, save_name),
                    aug_img
                )

                aug_index += 1
                current += 1





INPUT_DIR = "..\\Output\\training\\03_Contrast_Enhancement\\Useful_Insects"
OUTPUT_DIR = "..\\Output\\training\\04_data_balancing\\Useful_Insects"
os.makedirs(OUTPUT_DIR, exist_ok=True)

db = data_balancing()
db.process_folder(INPUT_DIR, OUTPUT_DIR)
print("\nDataset Balanced Successfully!")