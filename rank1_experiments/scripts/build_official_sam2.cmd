@echo off
call "C:\Users\CLEFFFF\anaconda3\condabin\conda.bat" activate "E:\4D_Gaussian\rank1_experiments\envs\morgs-cu121"
if errorlevel 1 exit /b %errorlevel%
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=E:\4D_Gaussian\rank1_experiments\envs\morgs-cu121
set TORCH_CUDA_ARCH_LIST=8.9
set MAX_JOBS=2
set DISTUTILS_USE_SDK=1
set PYTHONIOENCODING=utf-8
cd /d "E:\4D_Gaussian\rank1_experiments\build_sources_cu121\sam2-cuda"
python setup.py build_ext --build-lib "E:\4D_Gaussian\rank1_experiments\working\MoRGS"
exit /b %errorlevel%
