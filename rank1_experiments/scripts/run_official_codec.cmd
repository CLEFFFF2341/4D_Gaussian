@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
call "E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd" "E:\4D_Gaussian\rank1_experiments\scripts\phase01_official_ops_smoke.py" codec
exit /b %errorlevel%
