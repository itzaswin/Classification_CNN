import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA


class CCAInverseTangent:

    def __init__(
        self,
        analysis_csv,
        extraction_csv,
        output_folder,
        n_components=10
    ):

        self.analysis_csv = analysis_csv
        self.extraction_csv = extraction_csv
        self.output_folder = output_folder
        self.n_components = n_components

    #####################################################################

    def load_dataset(self):

        print("Loading datasets...")

        analysis = pd.read_csv(self.analysis_csv)
        extraction = pd.read_csv(self.extraction_csv)

        print("Analysis Shape   :", analysis.shape)
        print("Extraction Shape :", extraction.shape)

        required_columns = ["Image", "Class"]

        for col in required_columns:

            if col not in analysis.columns:
                raise ValueError(f"{col} not found in analysis CSV.")

            if col not in extraction.columns:
                raise ValueError(f"{col} not found in extraction CSV.")

        merged = pd.merge(
            analysis,
            extraction,
            on=["Image", "Class"],
            how="inner",
            suffixes=("_hog", "_dino")
        )

        if len(merged) == 0:
            raise ValueError("No matching images found.")

        self.metadata = merged[["Image", "Class"]]

        hog_columns = [c for c in merged.columns if c.endswith("_hog")]
        dino_columns = [c for c in merged.columns if c.endswith("_dino")]

        self.X = merged[hog_columns].astype(np.float32)
        self.Y = merged[dino_columns].astype(np.float32)

        print("\nMatched Images :", len(merged))
        print("HOG Features   :", self.X.shape)
        print("DINO Features  :", self.Y.shape)

    #####################################################################

    def preprocessing(self):

        X = self.X.values
        Y = self.Y.values

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        Y = np.nan_to_num(
            Y,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X = scaler_x.fit_transform(X)
        Y = scaler_y.fit_transform(Y)

        return X.astype(np.float32), Y.astype(np.float32)

    #####################################################################

    def canonical_correlation(self, X, Y):

        max_components = min(
            self.n_components,
            X.shape[0],
            X.shape[1],
            Y.shape[1]
        )

        print("\nCCA Components :", max_components)

        cca = CCA(
            n_components=max_components,
            max_iter=1000
        )

        X_c, Y_c = cca.fit_transform(X, Y)

        return X_c.astype(np.float32), Y_c.astype(np.float32)

    #####################################################################

    def inverse_tangent(self, X_c, Y_c):

        X_it = np.arctan(X_c).astype(np.float32)

        Y_it = np.arctan(Y_c).astype(np.float32)

        fused = np.concatenate(
            [X_it, Y_it],
            axis=1
        )

        return fused

    #####################################################################

    def save_features(self, fused):

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )

        df = self.metadata.copy()

        for i in range(fused.shape[1]):

            df[f"CCA_IT_{i+1}"] = fused[:, i]

        save_path = os.path.join(
            self.output_folder,
            "cca_it_features.csv"
        )

        df.to_csv(
            save_path,
            index=False
        )

        print("\n====================================")
        print("Feature Correlation Completed")
        print("Samples  :", len(df))
        print("Features :", fused.shape[1])
        print("Saved    :", save_path)
        print("====================================")

    #####################################################################

    def run(self):

        self.load_dataset()

        X, Y = self.preprocessing()

        X_c, Y_c = self.canonical_correlation(
            X,
            Y
        )

        fused = self.inverse_tangent(
            X_c,
            Y_c
        )

        self.save_features(fused)

        return fused


#########################################################################

if __name__ == "__main__":

    analysis_csv = "..\\Output\\training\\last_file\\feature_analysis.csv"

    extraction_csv = "..\\Output\\training\\last_file\\feature_extraction.csv"

    output_folder = "..\Output\training\06_csv_data"

    cca = CCAInverseTangent(

        analysis_csv=analysis_csv,

        extraction_csv=extraction_csv,

        output_folder=output_folder,

        n_components=10

    )

    features = cca.run()

    print("\nFinal Feature Shape :", features.shape)