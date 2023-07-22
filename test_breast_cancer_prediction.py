import unittest
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class BreastCancerPredictionTests(unittest.TestCase):

    def setUp(self):
        breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

        self.data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
        self.data_frame['label'] = breast_cancer_dataset.target

        self.X = self.data_frame.drop(columns='label', axis=1)
        self.Y = self.data_frame['label']

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=2)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train, self.Y_train)

    # Unittest
    def test_data_preprocessing(self): 
        self.assertEqual(self.data_frame.shape[0], 569)
        self.assertEqual(self.data_frame.shape[1], 31)

    # Unittest
    def test_model_training(self):
        self.assertIsNotNone(self.model)

    # Unittest
    def test_accuracy_on_training_data(self):
        X_train_prediction = self.model.predict(self.X_train)
        training_data_accuracy = accuracy_score(self.Y_train, X_train_prediction)
        self.assertGreaterEqual(training_data_accuracy, 0.0)
        self.assertLessEqual(training_data_accuracy, 1.0)

    # Unittest
    def test_accuracy_on_test_data(self):
        X_test_prediction = self.model.predict(self.X_test)
        test_data_accuracy = accuracy_score(self.Y_test, X_test_prediction)
        self.assertGreaterEqual(test_data_accuracy, 0.0)
        self.assertLessEqual(test_data_accuracy, 1.0)

    # Unittest
    def test_prediction(self):
        input_data = (15.34, 14.26, 102.5, 704.4, 0.1073, 0.2135, 0.2077, 0.09756, 0.2521, 0.07032,
                      0.4388, 0.7096, 3.384, 44.91, 0.006789, 0.05328, 0.06446, 0.02252, 0.03672,
                      0.004394, 18.07, 19.08, 125.1, 980.9, 0.139, 0.5954, 0.6305, 0.2393, 0.4667, 0.09946)

        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = self.model.predict(input_data_reshaped)
        self.assertIn(prediction[0], [0, 1])

if __name__ == '__main__':
    unittest.main()
