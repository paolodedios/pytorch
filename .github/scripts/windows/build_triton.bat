@echo on

set PYTHON_PREFIX=%PY_VERS:.=%
set PYTHON_PREFIX=py%PYTHON_PREFIX:;=;py%
call .ci/pytorch/win-test-helpers/installation-helpers/activate_miniconda3.bat
:: Create a new conda environment
if "%PY_VERS%" == "3.13t" (
    call conda create -n %PYTHON_PREFIX% -y -c=conda-forge python-freethreading python=3.13
) else (
    call conda create -n %PYTHON_PREFIX% -y -c=conda-forge python=%PY_VERS%
)
call conda run -n %PYTHON_PREFIX% pip install wheel pybind11 certifi cython cmake setuptools ninja

if not exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    echo Visual Studio %VC_YEAR% C++ BuildTools is required to compile PyTorch on Windows
    exit /b 1
)
set VC_VERSION_LOWER=17
set VC_VERSION_UPPER=18
if "%VC_YEAR%" == "2019" (
    set VC_VERSION_LOWER=16
    set VC_VERSION_UPPER=17
)
if NOT "%VS15INSTALLDIR%" == "" if exist "%VS15INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VS15VCVARSALL=%VS15INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat"
    goto vswhere
)

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -legacy -products * -version [%VC_VERSION_LOWER%^,%VC_VERSION_UPPER%^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto vswhere
    )
)

:vswhere
IF "%VS15VCVARSALL%"=="" (
    echo Visual Studio %VC_YEAR% C++ BuildTools is required to compile Triton on Windows
    exit /b 1
)
set MSSdk=1
set DISTUTILS_USE_SDK=1

call "%VS15VCVARSALL%" x64
call conda run -n %PYTHON_PREFIX% python .github/scripts/build_triton_wheel.py --device=%BUILD_DEVICE% %RELEASE%
