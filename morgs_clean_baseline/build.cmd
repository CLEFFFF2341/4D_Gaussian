@echo off
call "C:\Users\CLEFFFF\anaconda3\condabin\conda.bat" activate "%~dp0..\rank1_experiments\envs\morgs-cu121"
if errorlevel 1 exit /b %errorlevel%
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set CUDA_HOME=%CONDA_PREFIX%
set TORCH_CUDA_ARCH_LIST=8.9
set MAX_JOBS=2
set DISTUTILS_USE_SDK=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0native"
python setup.py build_ext --inplace
exit /b %errorlevel%
