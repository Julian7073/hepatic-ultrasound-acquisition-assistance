@echo off
setlocal
set "TESIS_ROOT=%~dp0.."
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%PYTHON_EXE%" (
    echo No se encontro Python en: "%PYTHON_EXE%"
    echo Verifique el interprete seleccionado en Visual Studio.
    pause
    exit /b 1
)
cd /d "%TESIS_ROOT%"
"%PYTHON_EXE%" -m streamlit run "%~dp0gui_adquisicion_hepatica.py" --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
endlocal