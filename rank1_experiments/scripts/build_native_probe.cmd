@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set TORCH_CUDA_ARCH_LIST=8.9
set MAX_JOBS=2
set DISTUTILS_USE_SDK=1
set PYTHONIOENCODING=utf-8
"E:\4D_Gaussian\rank1_experiments\envs\morgs-native-probe\Scripts\python.exe" -m pip wheel --no-deps --no-build-isolation "E:\4D_Gaussian\rank1_experiments\build_sources\gaussian-rasterization-grad" --wheel-dir "E:\4D_Gaussian\rank1_experiments\wheels"
exit /b %errorlevel%
