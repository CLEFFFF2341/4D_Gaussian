@echo off
call "C:\Users\CLEFFFF\anaconda3\condabin\conda.bat" activate "E:\4D_Gaussian\rank1_experiments\envs\morgs-cu121"
if errorlevel 1 exit /b %errorlevel%
set PYTHONIOENCODING=utf-8
set CUDA_HOME=E:\4D_Gaussian\rank1_experiments\envs\morgs-cu121
set TORCH_HOME=E:\4D_Gaussian\rank1_experiments\cache\torch
set TORCH_EXTENSIONS_DIR=E:\4D_Gaussian\rank1_experiments\cache\torch_extensions
set MAX_JOBS=2
cd /d "E:\4D_Gaussian\rank1_experiments\working\MoRGS"
python %*
exit /b %errorlevel%
