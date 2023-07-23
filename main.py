from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# load breast cancer dataset 
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# preprocess the breast cancer dataset
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

# set X and Y variables
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# train variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# set the model 

model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# Function to make a prediction
def make_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    return model.predict(input_data_reshaped)[0]

# front end program
@app.route('/', methods=['GET', 'POST'])
def index():
    training_data_accuracy = accuracy_score(Y_train, model.predict(X_train))
    test_data_accuracy = accuracy_score(Y_test, model.predict(X_test))
    

    if request.method == 'POST':
        input_data = [float(x) for x in request.form['input_data'].split(',')]
        prediction = make_prediction(input_data)
        if prediction == 0:
            result = 'Malignant'
        else:
            result = 'Benign'
        return render_template('index.html', prediction_result=result, training_accuracy=training_data_accuracy, test_accuracy=test_data_accuracy)

    return render_template('index.html', training_accuracy=training_data_accuracy, test_accuracy=test_data_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
