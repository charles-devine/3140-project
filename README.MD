## MyBreastHealth ##
hfaieahgiaehaighei

### Static Code Analysis: ###

OUTPUT

[Done] exited with code=0 in 1.357 seconds

[Running] python -u "c:\Users\charl\Desktop\3140-project\static_code_analysis.py"

************* Module main
main.py:1:0: C0114: Missing module docstring (missing-module-docstring)
main.py:12:26: E1101: Instance of 'tuple' has no 'data' member (no-member)
main.py:12:62: E1101: Instance of 'tuple' has no 'feature_names' member (no-member)
main.py:13:22: E1101: Instance of 'tuple' has no 'target' member (no-member)

------------------------------------------------------------------
Your code has been rated at 4.67/10

[Done] exited with code=0 in 5.782 seconds

### Dynamic Code Analysis: ###

################################################
Accuracy on training data = 0.9582417582417583
Accuracy on test data = 0.9385964912280702
Sample data = (15.34, 14.26, 102.5, 704.4, 0.1073, 0.2135, 0.2077, 0.09756, 0.2521, 0.07032, 0.4388, 0.7096, 3.384, 44.91, 0.006789, 0.05328, 0.06446, 0.02252, 0.03672, 0.004394, 18.07, 19.08, 125.1, 980.9, 0.139, 0.5954, 0.6305, 0.2393, 0.4667, 0.09946)
C:\Users\charl\AppData\Local\Programs\Python\Python39\lib\site-packages\sklearn\base.py:439: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
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


