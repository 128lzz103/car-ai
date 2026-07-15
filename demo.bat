@echo off
REM demo.bat - 项目演示脚本（Windows版本，无需API Key）

echo ==========================================
echo Minicoder Vehicle AI Demo
echo ==========================================
echo.

echo [Demo 1] Simple Range Query
echo Input: Check A102 range
echo ---
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '查询A102的续航']; main()"
echo.
echo.

echo [Demo 2] Missing Slot Detection
echo Input: Trip to Shanghai tomorrow, need charge?
echo ---
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '明天去上海，需要充电吗']; main()"
echo.
echo.

echo [Demo 3] Complete Trip Planning
echo Input: A102 to Shanghai Hongqiao at 8am, need charge?
echo ---
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', 'A102明早8点去上海虹桥站，判断是否需要充电']; main()"
echo.
echo.

echo [Demo 4] Knowledge Query
echo Input: How to maintain battery in winter
echo ---
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '冬天如何保养电池']; main()"
echo.
echo.

echo [Demo 5] Climate Control
echo Input: Set A102 AC to 24 degrees
echo ---
python -c "from minicoder.cli import main; import sys; sys.argv = ['minicoder', '--analyze-intent', '把A102的空调设为24度']; main()"
echo.
echo.

echo ==========================================
echo Demo Complete!
echo.
echo Core Capabilities:
echo   * Intent Recognition (7 vehicle intents)
echo   * Entity Extraction (vehicle_id, location, time, temp)
echo   * Slot Management (missing field detection)
echo   * Task Planning (6-step trip charge planning)
echo.
echo Mock Vehicle API: http://127.0.0.1:8765/docs
echo ==========================================
pause
