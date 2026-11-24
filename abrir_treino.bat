@echo off
set "ROOT=%~dp0.."
"%ROOT%\.venv_labelme\Scripts\labelme.exe" "%ROOT%\datast_Novo_cinza\train\images" --output "%~dp0train_json" --labels "%~dp0classes.txt"
