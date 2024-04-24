import joblib 
import numpy  as np
import pandas as pd

class CustomNaiveBayes:
    def __init__(self):
        self.class_probs = None
        self.feature_probs = None
        self.unique_classes = None

    def fit(self, X, y):
        # Calculate class probabilities
        self.class_probs = self.calculate_class_probs(y)
        
        # Calculate feature probabilities for each class
        self.feature_probs = self.calculate_feature_probs(X, y)

        # Get unique classes in the dataset
        self.unique_classes = np.unique(y)

    def calculate_class_probs(self, y):
        class_probs = {}
        total_samples = len(y)

        for cls in np.unique(y):
            class_probs[cls] = np.sum(y == cls) / total_samples

        return class_probs

    def calculate_feature_probs(self, X, y):
        num_features = X.shape[1]
        feature_probs = {}

        for cls in np.unique(y):
            class_mask = (y == cls)
            feature_probs[cls] = []

            for feature_index in range(num_features):
                feature_values = X[class_mask, feature_index]
                unique_values, counts = np.unique(feature_values, return_counts=True)

                # Add Laplace smoothing to handle zero probabilities
                smoothed_probs = (counts + 1) / (len(class_mask) + len(unique_values))
                feature_probs[cls].append((unique_values, smoothed_probs))

        return feature_probs

    def predict(self, X):
        predictions = []
        
        for sample in X:
            probs = []

            for cls in self.unique_classes:
                # Calculate the likelihood of the sample belonging to each class
                likelihood = 1.0

                for feature_index, feature_value in enumerate(sample):
                    unique_values, feature_probs = self.feature_probs[cls][feature_index]

                    # If the value is not present in training, use a small probability
                    if feature_value in unique_values:
                        likelihood *= feature_probs[np.where(unique_values == feature_value)[0][0]]
                    else:
                        likelihood *= 1e-5  # Small probability for unseen values

                # Multiply by class probability
                probs.append(self.class_probs[cls] * likelihood)

            # Choose the class with the highest probability
            predictions.append(self.unique_classes[np.argmax(probs)])

        return predictions
    




    def predict_proba(self, X):
        # Predict probabilities instead of classes
        probabilities = []

        for sample in X:
            probs = []

            for cls in self.unique_classes:
                # Calculate the likelihood of the sample belonging to each class
                likelihood = 1.0

                for feature_index, feature_value in enumerate(sample):
                    unique_values, feature_probs = self.feature_probs[cls][feature_index]

                    # If the value is not present in training, use a small probability
                    if feature_value in unique_values:
                        likelihood *= feature_probs[np.where(unique_values == feature_value)[0][0]]
                    else:
                        likelihood *= 1e-5  # Small probability for unseen values

                # Multiply by class probability
                probs.append(self.class_probs[cls] * likelihood)

            # Normalize probabilities to sum to 1
            probs /= np.sum(probs)
            probabilities.append(probs)




