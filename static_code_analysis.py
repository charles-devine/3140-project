import subprocess
file_to_analyze = "breast_cancer_prediction.py"
subprocess.run(["pylint", file_to_analyze])

# [Done] exited with code=0 in 1.357 seconds

# [Running] python -u "c:\Users\charl\Desktop\3140-project\static_code_analysis.py"

# ************* Module breast_cancer_prediction
# breast_cancer_prediction.py:1:0: C0114: Missing module docstring (missing-module-docstring)
# breast_cancer_prediction.py:12:26: E1101: Instance of 'tuple' has no 'data' member (no-member)
# breast_cancer_prediction.py:12:62: E1101: Instance of 'tuple' has no 'feature_names' member (no-member)
# breast_cancer_prediction.py:13:22: E1101: Instance of 'tuple' has no 'target' member (no-member)

# ------------------------------------------------------------------
# Your code has been rated at 4.67/10

# [Done] exited with code=0 in 5.782 seconds