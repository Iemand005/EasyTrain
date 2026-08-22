@echo off
./clean.bat
if exist ".\adapters\" rmdir /s /q ".\adapters"
pip install --force-reinstall --no-cache-dir -r requirements.txt