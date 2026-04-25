import subprocess, sys, os
os.chdir("/home/amluser/amlSettings/SLM/pairwise")
logfile = "../logs/pairwise_nohup.log"
os.makedirs("../logs", exist_ok=True)
with open(logfile, "w") as lf:
    p = subprocess.Popen(
        ["bash", "run_full.sh"],
        stdout=lf, stderr=subprocess.STDOUT,
        cwd="/home/amluser/amlSettings/SLM/pairwise"
    )
print(f"Started PID={p.pid}, log={logfile}")
