@echo off
./clean.bat
if exist ".\hf_cache\" rmdir /s /q ".\hf_cache"
if exist ".\adapters\" rmdir /s /q ".\adapters"
pip install --force-reinstall --no-cache-dir -r requirements.txt
