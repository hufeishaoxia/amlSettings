import ctypes
import time
import sys
import socketserver
import os
import argparse
import subprocess
from datetime import datetime
from threading import Thread

def get_environment_variable(name):
    try:
        return os.environ[name]
    except Exception:
        sys.stderr.write("Unable to find environment variable {var}".format(var=name))

class DummyTCPServer(socketserver.BaseRequestHandler):
    def handle(self):
        return

def run_once(local_stdout_log_file_path, local_stderr_log_file_path):
    args = sys.argv[1:]; 
    cmd = "{} > {} 2> {}".format(" ".join(args), local_stdout_log_file_path, local_stderr_log_file_path)
    print("Run '{}'".format(cmd));
    exit_code = os.system(cmd)
    print("Exit code = {}".format(exit_code));

def is_succeeded(output_file_path, local_stderr_log_file_path):
    if not os.path.exists(output_file_path):
        return False

    if os.path.getsize(output_file_path) == 0:
        return False

    if os.path.exists(local_stderr_log_file_path):
        with open(local_stderr_log_file_path) as f:
            for line in f:
                if (line.startswith("DLERROR:")):
                    return False;

    return True;

def run():
    input_file_path = str(get_environment_variable("_InputFilePath_"))
    output_file_path = str(get_environment_variable("_OutputFilePath_"))
    local_stdout_log_file_path = str(get_environment_variable("_StdoutLogFilePath_"))
    local_stderr_log_file_path = str(get_environment_variable("_StderrLogFilePath_"))
    complete_file_path = str(get_environment_variable("_CompleteFilePath_"))
    errors_file_path = str(get_environment_variable("_ErrorsFilePath_"))

    if os.path.exists(complete_file_path):
        print("Already completed before. {} exists".format(complete_file_path))
        return

    succeeded = False
    for i in range(3):
        try:
            run_once(local_stdout_log_file_path, local_stderr_log_file_path)
            if is_succeeded(output_file_path, local_stderr_log_file_path):
                succeeded = True
                break
        except Exception as e:            
            errMessage = str(e)
            print(errMessage)

        time.sleep(10)

    if not succeeded:
        print("Failed to run offline job")
        open(errors_file_path, "w")
    
    open(complete_file_path, "w")
    print("Done with offline job")

if __name__ == "__main__":

    print("Starting TCP server on new thread...")
    listening_port = int(get_environment_variable("_ListeningPort_"))
    server = socketserver.TCPServer(("localhost", listening_port), DummyTCPServer)
    serve_forever_thread = Thread(target=server.serve_forever)
    serve_forever_thread.start()

    run()

    serve_forever_thread.join()