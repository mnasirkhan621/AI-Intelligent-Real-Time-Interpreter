@echo off
echo ===================================================
echo   AI Real-Time Interpreter - Build Script
echo ===================================================
echo.
echo Installing PyInstaller...
pip install pyinstaller
echo.
echo Building Executable...
echo This may take a minute.
echo.
pyinstaller --noconfirm --onefile --windowed --name "AI_Interpreter" --icon=NONE --collect-all customtkinter --hidden-import "babel.numbers" --hidden-import "webrtcvad" main.py
echo.
echo ===================================================
echo   BUILD COMPLETE!
echo   You can find your app in the 'dist' folder.
echo ===================================================
pause
