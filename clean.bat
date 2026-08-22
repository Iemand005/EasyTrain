@echo off
if exist ".\reflex.lock\bun.lock" del ".\reflex.lock\bun.lock"
if exist ".\.reflex_uploads\" rmdir /s /q ".\.reflex_uploads"
if exist ".\.states\" rmdir /s /q ".\.states"
if exist ".\.web\" rmdir /s /q ".\.web"
if exist ".\uploaded_files\" rmdir /s /q ".\uploaded_files"
