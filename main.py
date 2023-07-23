from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the breast cancer dataset from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# Function to make a prediction
def make_prediction(input_data):
    # Convert the input string data to a list of floats
    input_data_as_list = [float(val) for val in input_data.split(",")]
    input_data_as_numpy_array = np.asarray(input_data_as_list)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    return model.predict(input_data_reshaped)[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = request.form['input_data']
        prediction = make_prediction(input_data)
        if prediction == 0:
            result = 'Malignant'
        else:
            result = 'Benign'
        return render_template('index.html', prediction_result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
