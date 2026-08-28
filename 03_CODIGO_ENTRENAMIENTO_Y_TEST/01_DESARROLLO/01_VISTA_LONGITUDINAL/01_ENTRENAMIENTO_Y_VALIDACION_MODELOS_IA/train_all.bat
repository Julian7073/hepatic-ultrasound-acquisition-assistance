@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set EPOCHS=30
set BATCH_SIZE=4
set IMAGE_SIZE=512

echo Proyecto: %CD%
echo EPOCHS=%EPOCHS% BATCH_SIZE=%BATCH_SIZE% IMAGE_SIZE=%IMAGE_SIZE%

echo.
echo ==============================
echo U-Net ROI/Higado/LA
echo ==============================
python ".\Codigos_Entrenamiento_Segmentacion\02_entrenar_unet.py" --class_name ROI --epochs %EPOCHS% --batch_size %BATCH_SIZE% --image_size %IMAGE_SIZE%
python ".\Codigos_Entrenamiento_Segmentacion\02_entrenar_unet.py" --class_name Higado --epochs %EPOCHS% --batch_size %BATCH_SIZE% --image_size %IMAGE_SIZE%
python ".\Codigos_Entrenamiento_Segmentacion\02_entrenar_unet.py" --class_name LA --epochs %EPOCHS% --batch_size %BATCH_SIZE% --image_size %IMAGE_SIZE%

echo.
echo ==============================
echo DeepLabV3+ ROI/Higado/LA
echo ==============================
python ".\Codigos_Entrenamiento_Segmentacion\03_entrenar_deeplabv3.py" --class_name ROI --epochs %EPOCHS% --batch_size %BATCH_SIZE% --image_size %IMAGE_SIZE%
python ".\Codigos_Entrenamiento_Segmentacion\03_entrenar_deeplabv3.py" --class_name Higado --epochs %EPOCHS% --batch_size %BATCH_SIZE% --image_size %IMAGE_SIZE%
python ".\Codigos_Entrenamiento_Segmentacion\03_entrenar_deeplabv3.py" --class_name LA --epochs %EPOCHS% --batch_size %BATCH_SIZE% --image_size %IMAGE_SIZE%

echo.
echo ==============================
echo SegFormer ROI/Higado/LA
echo ==============================
python ".\Codigos_Entrenamiento_Segmentacion\04_entrenar_segformer.py" --class_name ROI --epochs %EPOCHS% --batch_size %BATCH_SIZE% --image_size %IMAGE_SIZE%
python ".\Codigos_Entrenamiento_Segmentacion\04_entrenar_segformer.py" --class_name Higado --epochs %EPOCHS% --batch_size %BATCH_SIZE% --image_size %IMAGE_SIZE%
python ".\Codigos_Entrenamiento_Segmentacion\04_entrenar_segformer.py" --class_name LA --epochs %EPOCHS% --batch_size %BATCH_SIZE% --image_size %IMAGE_SIZE%

echo.
echo ==============================
echo Evaluacion y comparacion
echo ==============================
python ".\Codigos_Entrenamiento_Segmentacion\05_evaluar_modelos.py"
python ".\Codigos_Entrenamiento_Segmentacion\06_comparar_arquitecturas.py"
python ".\Codigos_Entrenamiento_Segmentacion\07_generar_reporte_segmentacion.py"

echo.
echo Entrenamiento completo.
pause
