    import os
    import json
    import joblib
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        ConfusionMatrixDisplay,
    )

    from tensorflow.keras import models, layers
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )


    class CNNClassifier:

        def __init__(
            self,
            file_paths,
            id_col,
            target_col,
            model_path="../Models/CNN.keras",
            scaler_path="../Models/scaler.pkl",
            encoder_path="../Models/label_encoder.pkl",
        ):

            self.file_paths = file_paths
            self.id_col = id_col
            self.target_col = target_col

            self.model_path = model_path
            self.scaler_path = scaler_path
            self.encoder_path = encoder_path

            self.model = None
            self.scaler = StandardScaler()
            self.encoder = LabelEncoder()

            self.history = None

            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None

            self.num_features = None
            self.num_classes = None

        # ============================================================
        # Load and Merge CSV Files
        # ============================================================

        def load_merge_dataset(self):

            print("Loading CSV files...")

            dataframes = []

            for file in self.file_paths:

                df = pd.read_csv(file)

                dataframes.append(df)

            merged = dataframes[0]

            for df in dataframes[1:]:

                merged = merged.merge(df, on=self.id_col)

            print("Merged Dataset Shape :", merged.shape)

            X = merged.drop(
                columns=[self.id_col, self.target_col]
            ).values

            y = merged[self.target_col].values

            return X, y

        # ============================================================
        # Data Preparation
        # ============================================================

        def prepare_data(self, test_size=0.20):

            X, y = self.load_merge_dataset()

            print("Encoding Labels...")

            y = self.encoder.fit_transform(y)

            os.makedirs("../Models", exist_ok=True)

            joblib.dump(
                self.encoder,
                self.encoder_path
            )

            print("Splitting Dataset...")

            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y,
            )

            print("Scaling Features...")

            X_train = self.scaler.fit_transform(X_train)

            X_test = self.scaler.transform(X_test)

            joblib.dump(
                self.scaler,
                self.scaler_path
            )

            self.X_train = np.expand_dims(
                X_train,
                axis=-1
            )

            self.X_test = np.expand_dims(
                X_test,
                axis=-1
            )

            self.y_train = y_train
            self.y_test = y_test

            self.num_features = self.X_train.shape[1]

            self.num_classes = len(
                np.unique(self.y_train)
            )

            print("-" * 50)
            print("Training Samples :", len(self.X_train))
            print("Testing Samples  :", len(self.X_test))
            print("Features         :", self.num_features)
            print("Classes          :", self.num_classes)
            print("-" * 50)

        # ============================================================
        # Build CNN Model
        # ============================================================

        def build_model(self):

            self.model = models.Sequential([

                layers.Input(
                    shape=(self.num_features, 1)
                ),

                layers.Conv1D(
                    filters=32,
                    kernel_size=3,
                    activation="relu",
                    padding="same"
                ),

                layers.BatchNormalization(),

                layers.MaxPooling1D(2),

                layers.Conv1D(
                    filters=64,
                    kernel_size=3,
                    activation="relu",
                    padding="same"
                ),

                layers.BatchNormalization(),

                layers.MaxPooling1D(2),

                layers.Conv1D(
                    filters=128,
                    kernel_size=3,
                    activation="relu",
                    padding="same"
                ),

                layers.GlobalAveragePooling1D(),

                layers.Dense(
                    128,
                    activation="relu"
                ),

                layers.Dropout(0.4),

                layers.Dense(
                    64,
                    activation="relu"
                ),

                layers.Dropout(0.3),

                layers.Dense(
                    self.num_classes,
                    activation="softmax"
                )

            ])

            self.model.compile(

                optimizer="adam",

                loss="sparse_categorical_crossentropy",

                metrics=["accuracy"]

            )

            print(self.model.summary())



        # ============================================================
        # Train CNN Model
        # ============================================================

        def train_model(self, epochs=30, batch_size=32):

            if self.model is None:
                raise ValueError("Build the CNN model before training.")

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1
            )

            checkpoint = ModelCheckpoint(
                filepath=self.model_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            )

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                verbose=1,
                min_lr=1e-6
            )

            self.history = self.model.fit(

                self.X_train,
                self.y_train,

                validation_data=(
                    self.X_test,
                    self.y_test
                ),

                epochs=epochs,

                batch_size=batch_size,

                callbacks=[
                    early_stop,
                    checkpoint,
                    reduce_lr
                ],

                verbose=1

            )

            print("Training Completed.")

        # ============================================================
        # Evaluate Model
        # ============================================================

        def evaluate_model(self):

            if self.model is None:
                raise ValueError("Model not available.")

            loss, accuracy = self.model.evaluate(

                self.X_test,
                self.y_test,
                verbose=0

            )

            print("-" * 50)
            print(f"Test Loss     : {loss:.4f}")
            print(f"Test Accuracy : {accuracy:.4f}")
            print("-" * 50)

            return loss, accuracy


        # ============================================================
        # Classification Report
        # ============================================================

        def classification_report(self):

            predictions = self.model.predict(
                self.X_test,
                verbose=0
            )

            predicted_labels = np.argmax(
                predictions,
                axis=1
            )

            names = self.encoder.classes_

            report = classification_report(

                self.y_test,

                predicted_labels,

                target_names=names

            )

            print(report)


        # ============================================================
        # Confusion Matrix
        # ============================================================

        def plot_confusion_matrix(self):

            predictions = self.model.predict(
                self.X_test,
                verbose=0
            )

            predicted_labels = np.argmax(
                predictions,
                axis=1
            )

            cm = confusion_matrix(

                self.y_test,

                predicted_labels

            )

            disp = ConfusionMatrixDisplay(

                confusion_matrix=cm,

                display_labels=self.encoder.classes_

            )

            plt.figure(figsize=(8, 8))

            disp.plot(
                cmap="Blues",
                values_format="d"
            )

            plt.title("Confusion Matrix")

            plt.show()


        # ============================================================
        # Training Graph
        # ============================================================

        def plot_training_history(self):

            if self.history is None:
                return

            history = self.history.history

            plt.figure(figsize=(8,5))

            plt.plot(
                history["accuracy"],
                label="Train Accuracy"
            )

            plt.plot(
                history["val_accuracy"],
                label="Validation Accuracy"
            )

            plt.xlabel("Epoch")

            plt.ylabel("Accuracy")

            plt.title("CNN Accuracy")

            plt.legend()

            plt.grid(True)

            plt.show()


            plt.figure(figsize=(8,5))

            plt.plot(
                history["loss"],
                label="Train Loss"
            )

            plt.plot(
                history["val_loss"],
                label="Validation Loss"
            )

            plt.xlabel("Epoch")

            plt.ylabel("Loss")

            plt.title("CNN Loss")

            plt.legend()

            plt.grid(True)

            plt.show()


        # ============================================================
        # Load Saved Model
        # ============================================================

        def load_model(self):

            self.model = tf.keras.models.load_model(
                self.model_path
            )

            self.scaler = joblib.load(
                self.scaler_path
            )

            self.encoder = joblib.load(
                self.encoder_path
            )

            print("Model Loaded Successfully.")


        # ============================================================
        # Predict Using CSV Feature Vector
        # ============================================================

        def predict(self, feature_vector):

            feature_vector = np.array(
                feature_vector
            ).reshape(1, -1)

            feature_vector = self.scaler.transform(
                feature_vector
            )

            feature_vector = np.expand_dims(
                feature_vector,
                axis=-1
            )

            prediction = self.model.predict(
                feature_vector,
                verbose=0
            )

            index = np.argmax(
                prediction,
                axis=1
            )[0]

            class_name = self.encoder.inverse_transform(
                [index]
            )[0]

            confidence = float(
                prediction[0][index]
            )

            print("-" * 50)
            print("Predicted Class :", class_name)
            print("Confidence      :", round(confidence * 100, 2), "%")
            print("-" * 50)

            return class_name, confidence



        # =====================================================
        # Rule Evaluation
        # =====================================================

        def evaluate_rules(
            self,
            confidence_membership,
            severity_membership
        ):

            rules = {}

            # Rule 1
            rules["Chemical"] = max(
                min(
                    confidence_membership["High"],
                    severity_membership["High"]
                ),
                min(
                    confidence_membership["High"],
                    severity_membership["Medium"]
                )
            )

            # Rule 2
            rules["Bio"] = max(
                min(
                    confidence_membership["Medium"],
                    severity_membership["Medium"]
                ),
                min(
                    confidence_membership["High"],
                    severity_membership["Low"]
                ),
                min(
                    confidence_membership["Medium"],
                    severity_membership["Low"]
                )
            )

            # Rule 3
            rules["Organic"] = max(
                min(
                    confidence_membership["Low"],
                    severity_membership["Low"]
                ),
                min(
                    confidence_membership["Low"],
                    severity_membership["Medium"]
                )
            )

            # Rule 4
            rules["ManualInspection"] = min(
                confidence_membership["Low"],
                severity_membership["High"]
            )

            return rules

        # =====================================================
        # Aggregate Rule Strength
        # =====================================================

        def aggregate_score(
            self,
            rules
        ):

            score = 0.0

            score += rules["Chemical"] * 90

            score += rules["Bio"] * 70

            score += rules["Organic"] * 50

            score += rules["ManualInspection"] * 20

            total = sum(rules.values())

            if total == 0:

                return 0

            return score / total

        # =====================================================
        # Find Matching Recommendation
        # =====================================================

        def search_recommendation(
            self,
            insect_name
        ):

            result = self.dataset[
                self.dataset["Insect"].str.lower()
                ==
                insect_name.lower()
            ]

            if result.empty:

                return None

            return result

        # =====================================================
        # Select Best Recommendation
        # =====================================================

        def select_best_recommendation(
            self,
            insect_name,
            confidence
        ):

            recommendation = self.search_recommendation(
                insect_name
            )

            if recommendation is None:

                return None

            confidence_membership = self.confidence_membership(
                confidence
            )

            recommendations = []

            for _, row in recommendation.iterrows():

                severity = str(row["Severity"])

                severity_membership = self.severity_membership(
                    severity
                )

                rules = self.evaluate_rules(
                    confidence_membership,
                    severity_membership
                )

                fuzzy_score = self.aggregate_score(
                    rules
                )

                recommendations.append({

                    "Insect": row["Insect"],

                    "Severity": row["Severity"],

                    "Pesticide": row["Pesticide"],

                    "Dosage": row["Dosage"],

                    "Score": fuzzy_score,

                    "Rules": rules

                })

            recommendations = sorted(

                recommendations,

                key=lambda x: x["Score"],

                reverse=True

            )

            return recommendations



        # =====================================================
        # Final Recommendation
        # =====================================================

        def recommend(
            self,
            insect_name,
            confidence
        ):

            recommendations = self.select_best_recommendation(
                insect_name,
                confidence
            )

            if recommendations is None:

                print("-------------------------------------")
                print("No Recommendation Found")
                print("-------------------------------------")

                return None

            best = recommendations[0]

            if best["Score"] >= 80:

                recommendation_level = "Very High"

            elif best["Score"] >= 60:

                recommendation_level = "High"

            elif best["Score"] >= 40:

                recommendation_level = "Medium"

            else:

                recommendation_level = "Low"

            result = {

                "Predicted Insect": best["Insect"],

                "CNN Confidence (%)": confidence,

                "Severity": best["Severity"],

                "Recommended Pesticide": best["Pesticide"],

                "Dosage": best["Dosage"],

                "Recommendation Score": round(
                    best["Score"],
                    2
                ),

                "Recommendation Level": recommendation_level

            }

            return result


        # =====================================================
        # Display Result
        # =====================================================

        def display_result(
            self,
            result
        ):

            if result is None:

                return

            print("\n")
            print("=" * 60)

            print("FINAL PESTICIDE RECOMMENDATION")

            print("=" * 60)

            print(f"Predicted Insect        : {result['Predicted Insect']}")

            print(f"CNN Confidence          : {result['CNN Confidence (%)']} %")

            print(f"Severity                : {result['Severity']}")

            print(f"Recommended Pesticide   : {result['Recommended Pesticide']}")

            print(f"Dosage                  : {result['Dosage']}")

            print(f"Recommendation Score    : {result['Recommendation Score']}")

            print(f"Recommendation Level    : {result['Recommendation Level']}")

            print("=" * 60)



        def predict(self,insect_name,confidence):

            result = self.recommend(insect_name,confidence)
            self.display_result(result)
            return result


    def run(self):
        print("\nStarting CNN Pipeline...\n")

        # 1. Prepare Dataset
        self.prepare_data()

        # 2. Build CNN Model
        self.build_model()

        # 3. Train Model
        self.train_model()

        # 4. Evaluate Model
        self.evaluate_model()

        # 5. Classification Report
        self.classification_report()

        # 6. Confusion Matrix
        self.plot_confusion_matrix()

        # 7. Training Graphs
        self.plot_training_history()

        print("\nCNN Pipeline Completed Successfully.")

#
# if __name__ == "__main__":
#
#     files = [
#         "../Dataset/features1.csv",
#         "../Dataset/features2.csv",
#         "../Dataset/features3.csv"
#     ]
#
#     cnn = CNNClassifier(
#         file_paths=files,
#         id_col="ID",
#         target_col="Insect"
#     )
#
#     cnn.run()