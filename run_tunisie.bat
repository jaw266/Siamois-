@echo off
chcp 65001 > nul
title PFE - Tests Tunisie HiveMQ

echo ============================================================
echo   PFE - Detection de changement - Images Tunisie
echo ============================================================
echo.

cd /d "%~dp0backend"

echo [1/2] Test Tunisie 03...
python inference_onnx_mqtt_hivemq.py ^
  --t1 ../test_images/test_03_tunisie/T1.png ^
  --t2 ../test_images/test_03_tunisie/T2.png ^
  --id_test test_03_tunisie

echo.
echo [2/2] Test Tunisie 04...
python inference_onnx_mqtt_hivemq.py ^
  --t1 ../test_images/test_04_tunisie/T1.png ^
  --t2 ../test_images/test_04_tunisie/T2.png ^
  --id_test test_04_tunisie

echo.
echo ============================================================
echo   Termine. Resultats publies sur HiveMQ Cloud.
echo ============================================================
pause
