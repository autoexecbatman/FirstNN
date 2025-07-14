@echo off
echo Copying PyTorch DLLs to CleanMNIST...
copy "D:\libtorch-cuda\libtorch\lib\*.dll" "D:\repo\firstNN\build\Debug\"
echo Done!
pause
