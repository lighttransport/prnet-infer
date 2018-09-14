rmdir /q /s build
mkdir build

rem source code directory of tensorflow for C
SET TF_C_DIR=C:\Users\syoyo\local\libtensorflow-cpu-windows-x86_64-1.10.0

cmake -G "Visual Studio 15 2017 Win64" ^
      -DTENSORFLOW_C_DIR=%TF_C_DIR% ^
      -DWITH_GUI=On ^
      -Bbuild ^
      -H.
