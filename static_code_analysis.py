import subprocess
file_to_analyze = "main.py"
subprocess.run(["pylint", file_to_analyze])

# OUTPUT

# [Done] exited with code=0 in 1.357 seconds

# [Running] python -u "c:\Users\charl\Desktop\3140-project\static_code_analysis.py"

# ************* Module main
# main.py:1:0: C0114: Missing module docstring (missing-module-docstring)
# main.py:12:26: E1101: Instance of 'tuple' has no 'data' member (no-member)
# main.py:12:62: E1101: Instance of 'tuple' has no 'feature_names' member (no-member)
# main.py:13:22: E1101: Instance of 'tuple' has no 'target' member (no-member)

# ------------------------------------------------------------------
# Your code has been rated at 4.67/10

# [Done] exited with code=0 in 5.782 seconds