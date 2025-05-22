@echo off
echo Starting all servers...

REM Start the main server
start cmd /k "cd D:\BookTeaser && python server_app\server_main.py"

REM Start the file uploader server
start cmd /k "cd D:\BookTeaser && python server_app\server_file_uploader.py"

REM Start the video local server
start cmd /k "cd D:\BookTeaser && python server_app\server_video_local.py"

REM Start the development server
start cmd /k "cd D:\BookTeaser\frontend_app && npm run dev"



echo Waiting for servers to initialize...
timeout /t 15 /nobreak

REM Start Chrome in fullscreen mode
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --kiosk --start-fullscreen --app="http://localhost:7778/#0.5" 
 
REM --incognito --disable-application-cache --disk-cache-size=1 --media-cache-size=1

echo All components have been started! 