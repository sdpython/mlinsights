@echo off
set current=%~dp0
set root=%current%..
cd %root%
@echo ##################
@echo Compile
@echo running %root%\setup.py build_ext --inplace
@echo ##################
set pythonexe="c:\Python370_x64\python.exe"
if not exist %pythonexe% set pythonexe="c:\Python366_x64\python.exe"
if not exist %pythonexe% set pythonexe="c:\Python365_x64\python.exe"
if not exist %pythonexe% set pythonexe="c:\Python364_x64\python.exe"
if not exist %pythonexe% set pythonexe="c:\Python363_x64\python.exe"
if not exist %pythonexe% set pythonexe="c:\Python36_x64\python.exe"
%pythonexe% %root%\setup.py build_ext --inplace
if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Compile.
@echo ##################
@echo Build
@echo running %root%\setup.py bdist_wheel
@echo ##################
%pythonexe% %root%\setup.py bdist_wheel
if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Build.
cd %current%