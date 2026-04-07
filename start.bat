@echo off
chcp 65001 >nul 2>&1
title EZ Transcript

echo ============================================
echo   EZ Transcript — Transcription locale IA
echo ============================================
echo.

:: Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Python n'est pas installé ou pas dans le PATH.
    echo Téléchargez Python 3.10+ depuis https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Python 3.10 ou supérieur est requis.
    python --version
    pause
    exit /b 1
)

:: Check FFmpeg
where ffmpeg >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [AVERTISSEMENT] FFmpeg n'est pas installé ou pas dans le PATH.
    echo L'application en a besoin pour traiter les fichiers audio/vidéo.
    echo Téléchargez-le depuis https://ffmpeg.org/download.html
    echo.
)

:: Create virtual environment if needed
if not exist "venv" (
    echo [INFO] Création de l'environnement virtuel...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERREUR] Impossible de créer l'environnement virtuel.
        pause
        exit /b 1
    )
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install/update dependencies
echo [INFO] Vérification des dépendances...
pip install -r requirements.txt --quiet 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installation des dépendances...
    pip install -r requirements.txt
)

:: Detect GPU
echo.
echo [INFO] Détection du matériel...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "tokens=1,2 delims=," %%a in ('nvidia-smi --query-gpu^=name^,memory.total --format^=csv^,noheader^,nounits') do (
        echo   GPU: %%a ^(%%b Mo VRAM^)
    )
    :: Install CUDA runtime libraries if not already present
    python -c "import ctypes; ctypes.WinDLL('cublas64_12.dll')" >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [INFO] Installation des bibliothèques CUDA...
        pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 --quiet
        if %ERRORLEVEL% equ 0 (
            echo [INFO] CUDA installé avec succès.
        ) else (
            echo [AVERTISSEMENT] Impossible d'installer CUDA — le mode CPU sera utilisé.
        )
    ) else (
        echo   CUDA: opérationnel
    )
) else (
    echo   Pas de GPU NVIDIA détecté — mode CPU
)

:: Set port
set PORT=7860

:: Launch server
echo.
echo [INFO] Démarrage du serveur sur http://localhost:%PORT%
echo [INFO] Appuyez sur Ctrl+C pour arrêter.
echo.

:: Open browser after a short delay
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:%PORT%"

:: Run server
python -m uvicorn backend.main:app --host 0.0.0.0 --port %PORT% --no-access-log

:: If server stops
echo.
echo [INFO] Serveur arrêté.
pause
