# MyBreastHealth

# Static and Dynamic Code Analysis

This repository contains a Python script `main.py` that performs breast cancer prediction using logistic regression. The script uses the breast cancer dataset from scikit-learn and trains a logistic regression model to predict whether a tumor is benign or malignant. Additionally, we have performed static and dynamic code analysis on the script to ensure its quality.

## Dynamic Code Analysis

To perform dynamic code analysis, we used the `coverage` tool, which measures the code coverage during test execution. Below is the output of running `coverage run` on the test script:

```plaintext
PS C:\Users\charl\Desktop\3140-project> coverage run test_main.py
c:\users\charl\appdata\local\programs\python\python39\lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
################################################
Accuracy on training data = 0.9582417582417583
Accuracy on test data = 0.9385964912280702
Sample data = (15.34, 14.26, 102.5, 704.4, 0.1073, 0.2135, 0.2077, 0.09756, 0.2521, 0.07032, 0.4388, 0.7096, 3.384, 44.91, 0.006789, 0.05328, 0.06446, 0.02252, 0.03672, 0.004394, 18.07, 19.08, 125.1, 980.9, 0.139, 0.5954, 0.6305, 0.2393, 0.4667, 0.09946)
c:\users\charl\appdata\local\programs\python\python39\lib\site-packages\sklearn\base.py:439: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
  warnings.warn(
Prediction: The Breast cancer is Malignant
################################################
Name      Stmts   Miss  Cover
-----------------------------
main.py      30      1    97%
-----------------------------
TOTAL        30      1    97%

----------------------------------------------------------------------
Ran 0 tests in 0.000s

OK
```

## Static Code Analysis

Static code analysis was performed using the `pylint tool`, which checks the code against various coding standards and best practices. Below is the output of running pylint on the script:

```plaintext
PS C:\Users\charl\Desktop\3140-project> python -u "c:\Users\charl\Desktop\3140-project\static_code_analysis.py"

************* Module main
main.py:1:0: C0114: Missing module docstring (missing-module-docstring)
main.py:12:26: E1101: Instance of 'tuple' has no 'data' member (no-member)
main.py:12:62: E1101: Instance of 'tuple' has no 'feature_names' member (no-member)
main.py:13:22: E1101: Instance of 'tuple' has no 'target' member (no-member)

------------------------------------------------------------------
Your code has been rated at 4.67/10
```
