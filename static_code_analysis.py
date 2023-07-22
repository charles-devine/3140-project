import subprocess
file_to_analyze = "main.py"
subprocess.run(["pylint", file_to_analyze])