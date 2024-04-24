import joblib
import pandas as pd
import numpy as np
class DecisionTreeID2:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        self.tree = self._grow_tree(data)

    def _grow_tree(self, data):
        if len(set(data.iloc[:, -1])) == 1:
            return {'class': data.iloc[0, -1]}

        if len(data.columns) == 1:
            return {'class': self._majority_class(data.iloc[:, -1])}

        best_feature = self._choose_best_feature(data)
        tree = {'feature': best_feature, 'branches': {}}

        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            tree['branches'][value] = self._grow_tree(subset.drop(columns=[best_feature]))

        return tree

    def _choose_best_feature(self, data):
        features = data.columns[:-1]
        info_gain = [self._information_gain(data, feature) for feature in features]
        return features[np.argmax(info_gain)]

    def _information_gain(self, data, feature):
        entropy_before = self._entropy(data.iloc[:, -1])
        values, counts = np.unique(data[feature], return_counts=True)

        entropy_after = sum((counts[i] / len(data)) * self._entropy(data[data[feature] == values[i]].iloc[:, -1]) for i in range(len(values)))

        return entropy_before - entropy_after

    def _entropy(self, target):
        values, counts = np.unique(target, return_counts=True)
        probabilities = counts / len(target)
        entropy = -sum(p * np.log2(p) for p in probabilities)
        return entropy

    def _majority_class(self, target):
        return target.mode().iloc[0]

    def predict(self, X):
        return [self._predict_one(x, self.tree) for _, x in X.iterrows()]

    def _predict_one(self, x, node):
        if 'class' in node:
            return node['class']

        feature_value = x[node['feature']]

        if feature_value in node['branches']:
            return self._predict_one(x, node['branches'][feature_value])
        else:
            # If the value is not in the training data, return the majority class
            return self._majority_class(pd.Series(node['branches']).apply(lambda x: x['class']))

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

        return np.array(probabilities)
    def save(self, filename):
        
        joblib.dump(self, filename)