import subprocess
from time import sleep
from os import path, remove
from settings import start_path

m_start_path = start_path + "\ProcessMonitor"
def recording():
    csv_path = m_start_path + r"\rec_syscalls.csv"
    pml_path = m_start_path + r"\rec_syscalls.pml"
    if path.exists(csv_path):
        remove(csv_path)
        remove(pml_path)
    subprocess.Popen([m_start_path + r"\Start-procman.bat"])
    while not (path.exists(csv_path)):
        print("recording...")
        sleep(5)
    sleep(15)
    return "exists"