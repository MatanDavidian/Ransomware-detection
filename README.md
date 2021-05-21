# installation
* install fast text library and add it to project directory:
https://medium.com/@oleg.tarasov/building-fasttext-python-wrapper-from-source-under-windows-68e693a68cbb
  
* download  fasttext model:
  https://fasttext.cc/docs/en/crawl-vectors.html
  
* change start_path in setting.py to project path.
* change Scan.bat
  * change the name of virtual env.
  * change path of: detect_malware.py to the right path.
* change path in Start-procman.bat to the path of ProcessMonitor folder.
# run
* run program from Scan.bat file **as admin** .
# description
the program detect ransomware viruses
according the syscalls that performed in the computer.  
the program record all syscalls that perform in the computer for 50(s),
and check any process that perform more than 1.8k syscalls
