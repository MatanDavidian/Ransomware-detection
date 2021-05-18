cd C:\Users\SCE\PycharmProjects\final_p\ProcessMonitor
@ECHO OFF
powershell -command "./Start-procman.ps1 -Filter \"Process Name,is,Procmon.exe,Exclude;Process Name,is,Procmon64.exe,Exclude\""
