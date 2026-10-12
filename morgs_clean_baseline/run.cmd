@echo off
call "C:\Users\CLEFFFF\anaconda3\condabin\conda.bat" activate "%~dp0..\rank1_experiments\envs\morgs-cu121"
if errorlevel 1 exit /b %errorlevel%
set PYTHONIOENCODING=utf-8
set TORCH_HOME=%~dp0..\rank1_experiments\cache\torch
cd /d "%~dp0"
python %*
exit /b %errorlevel%
