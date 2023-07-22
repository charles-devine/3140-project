import unittest
import coverage

cov = coverage.Coverage()
cov.start()

# Import your main script here
import main

if __name__ == '__main__':
    # Stop coverage measurement and generate the report after running the tests
    cov.stop()
    cov.save()
    cov.report()

    # Run the tests from the main script
    unittest.main()
    
# OUTPUT

# PS C:\Users\charl\Desktop\3140-project> coverage run test_main.py
# c:\users\charl\appdata\local\programs\python\python39\lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
# 
# Increase the number of iterations (max_iter) or scale the data as shown in:
#     https://scikit-learn.org/stable/modules/preprocessing.html
# Please also refer to the documentation for alternative solver options:
#     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#   n_iter_i = _check_optimize_result(
# ################################################
# Accuracy on training data = 0.9582417582417583
# Accuracy on test data = 0.9385964912280702
# Sample data = (15.34, 14.26, 102.5, 704.4, 0.1073, 0.2135, 0.2077, 0.09756, 0.2521, 0.07032, 0.4388, 0.7096, 3.384, 44.91, 0.006789, 0.05328, 0.06446, 0.02252, 0.03672, 0.004394, 18.07, 19.08, 125.1, 980.9, 0.139, 0.5954, 0.6305, 0.2393, 0.4667, 0.09946)    
# c:\users\charl\appdata\local\programs\python\python39\lib\site-packages\sklearn\base.py:439: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
#   warnings.warn(
# Prediction: The Breast cancer is Malignant
# ################################################
# Name      Stmts   Miss  Cover
# -----------------------------
# main.py      30      1    97%
# -----------------------------
# TOTAL        30      1    97%

# ----------------------------------------------------------------------
# Ran 0 tests in 0.000s

# OK