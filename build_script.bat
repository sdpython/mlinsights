@echo off
if "%1"=="" goto default_value_python:
set pythonexe="%1"
%pythonexe% setup.py write_version
goto custom_python:

:default_value_python:
set pythonexe="c:\Python395_x64\python.exe"
if not exist %pythonexe% set pythonexe="c:\Python391_x64\python.exe"
:custom_python:
@echo [python] %pythonexe%
%pythonexe% -u setup.py build_script
if %errorlevel% neq 0 exit /b %errorlevel%