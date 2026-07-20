import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


# ============================================================
# Gaussian Mixture Model
# ============================================================
class GMM:

    def __init__(self, n_components=2, max_iter=50, tol=1e-3, reg=1e-6):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg

    def _initialize(self, X):

        kmeans = KMeans(
            n_clusters=self.k,
            n_init=10,
            random_state=42
        )

        labels = kmeans.fit_predict(X)

        self.means = kmeans.cluster_centers_

        d = X.shape[1]

        self.covars = np.array([
            np.eye(d)
            for _ in range(self.k)
        ], dtype=float)

        self.weights = np.zeros(self.k)

        for i in range(self.k):
            self.weights[i] = np.mean(labels == i)

    def fit(self, X):

        X = np.asarray(X, dtype=float)

        n, d = X.shape

        self._initialize(X)

        prev_log_likelihood = -np.inf

        for _ in range(self.max_iter):

            responsibilities = np.zeros((n, self.k))

            # ---------------- E-Step ----------------

            for c in range(self.k):

                covariance = self.covars[c] + np.eye(d) * self.reg

                responsibilities[:, c] = (
                    self.weights[c]
                    * multivariate_normal.pdf(
                        X,
                        mean=self.means[c],
                        cov=covariance,
                        allow_singular=True
                    )
                )

            total = responsibilities.sum(axis=1, keepdims=True) + 1e-12

            log_likelihood = np.sum(np.log(total))

            responsibilities /= total

            Nk = responsibilities.sum(axis=0) + 1e-12

            # ---------------- M-Step ----------------

            self.weights = Nk / n

            self.means = (responsibilities.T @ X) / Nk[:, None]

            for c in range(self.k):

                diff = X - self.means[c]

                covariance = (
                    (responsibilities[:, c][:, None] * diff).T @ diff
                ) / Nk[c]

                covariance += np.eye(d) * self.reg

                self.covars[c] = covariance

            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break

            prev_log_likelihood = log_likelihood

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        n = X.shape[0]

        probabilities = np.zeros((n, self.k))

        for c in range(self.k):

            covariance = self.covars[c] + np.eye(X.shape[1]) * self.reg

            probabilities[:, c] = (
                self.weights[c]
                * multivariate_normal.pdf(
                    X,
                    mean=self.means[c],
                    cov=covariance,
                    allow_singular=True
                )
            )

        return np.argmax(probabilities, axis=1)


# ============================================================
# Background Removal
# ============================================================
class BackgroundRemoval:

    def __init__(self,
                 resize=(512, 512),
                 sample_size=5000):

        self.resize = resize
        self.sample_size = sample_size

    def process_image(self, input_path, output_path):

        image = cv2.imread(input_path)

        if image is None:
            raise ValueError(f"Cannot read image: {input_path}")

        image = cv2.resize(image, self.resize)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pixels = rgb.reshape(-1, 3).astype(np.float64)

        # ------------------------------------------
        # Faster Training
        # ------------------------------------------

        if len(pixels) > self.sample_size:

            indices = np.random.choice(
                len(pixels),
                self.sample_size,
                replace=False
            )

            sample = pixels[indices]

        else:

            sample = pixels

        gmm = GMM(n_components=2)

        gmm.fit(sample)

        labels = gmm.predict(pixels)

        labels = labels.reshape(image.shape[:2])

        values, counts = np.unique(labels, return_counts=True)

        background_cluster = values[np.argmax(counts)]

        mask = np.where(
            labels == background_cluster,
            0,
            255
        ).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        foreground = cv2.bitwise_and(
            image,
            image,
            mask=mask
        )

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(output_path, foreground)

        print(f"Saved : {output_path}")

        # Display

        foreground_rgb = cv2.cvtColor(
            foreground,
            cv2.COLOR_BGR2RGB
        )

        #plt.figure(figsize=(6, 6))
        #plt.imshow(foreground_rgb)
        #plt.title("Foreground Removal")
        #plt.axis("off")
        #plt.show()

        return output_path

    def process_folder(self, input_folder, output_folder):

        extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff"
        )

        total = 0

        for root, _, files in os.walk(input_folder):

            for file in files:

                if file.lower().endswith(extensions):

                    input_path = os.path.join(root, file)

                    relative = os.path.relpath(
                        root,
                        input_folder
                    )

                    save_folder = os.path.join(
                        output_folder,
                        relative
                    )

                    os.makedirs(
                        save_folder,
                        exist_ok=True
                    )

                    output_path = os.path.join(
                        save_folder,
                        file
                    )

                    try:

                        print(f"Processing : {input_path}, Count : {total}")

                        self.process_image(
                            input_path,
                            output_path
                        )

                        total += 1

                    except Exception as e:

                        print(f"Error : {input_path}")
                        print(e)

        print("-" * 50)
        print(f"Completed : {total} images")
        print("-" * 50)


# ============================================================
# Main
# ============================================================

input_folder = "..\\Output\\training\\04_data_balancing\\Harmful_Insects"

output_folder = "..\\Output\\training\\05_background_removal\\Harmful_Insects"

bg = BackgroundRemoval(
        resize=(512, 512)
    )

bg.process_folder(
        input_folder,
        output_folder
    )