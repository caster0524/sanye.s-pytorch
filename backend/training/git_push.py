import subprocess, os
os.chdir(r"c:\sanyepytorch\projects\pytorch-web-app")

log_path = r"c:\sanyepytorch\projects\pytorch-web-app\git_push_output.txt"
with open(log_path, "w") as f:
    result = subprocess.run(["git", "pull", "origin", "main", "--no-rebase"], capture_output=True, text=True)
    f.write("=== PULL ===\n")
    f.write(f"STDOUT: {result.stdout}\n")
    f.write(f"STDERR: {result.stderr}\n")
    f.write(f"RC: {result.returncode}\n\n")

    result2 = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
    f.write("=== PUSH ===\n")
    f.write(f"STDOUT: {result2.stdout}\n")
    f.write(f"STDERR: {result2.stderr}\n")
    f.write(f"RC: {result2.returncode}\n")
print("DONE")