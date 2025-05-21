@echo off
echo Starting all servers...


REM Start Chrome in fullscreen mode
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --kiosk --start-fullscreen --app="http://localhost:7778/#prod"

echo All components have been started! 