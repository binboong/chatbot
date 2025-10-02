@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title 🚀 Vietnamese Chatbot Launcher

set CONTAINER=chatbot_service
set IMAGE=chatbot-chatbot:latest

:menu
cls
echo ============================================
echo 🤖 Vietnamese Chatbot Launcher - Menu
echo ============================================
echo.
echo 1. 🔨 Build Docker image
echo 2. 🚀 Khởi động container
echo 3. 🧠 Update database (setup + ingest) cần phải khởi động lại container sau khi update
echo 4. 💬 Gửi prompt đến chatbot
echo 5. 📜 Xem log container
echo 6. 🔍 Kiểm tra trạng thái
echo 7. 🛑 Dừng và xóa container
echo 8. 🗑️ Xóa image
echo 9. 🧹 Clean toàn bộ (container + image + volumes)
echo 0. 🔄 Rebuild và restart
echo exit. ❌ Thoát chương trình
echo.
set /p choice=👉 Nhập lựa chọn của bạn: 

if "%choice%"=="1" goto build
if "%choice%"=="2" goto run
if "%choice%"=="3" goto update
if "%choice%"=="4" goto prompt
if "%choice%"=="5" goto logs
if "%choice%"=="6" goto status
if "%choice%"=="7" goto stop
if "%choice%"=="8" goto remove_image
if "%choice%"=="9" goto clean_all
if "%choice%"=="0" goto rebuild
if /i "%choice%"=="exit" goto end
echo ⚠️ Lựa chọn không hợp lệ!
timeout /t 2 >nul
goto menu

:build
echo.
echo 🔨 Đang build Docker image...
echo ℹ️ Tạo các thư mục cần thiết nếu chưa có...
if not exist "data" mkdir data
if not exist "vector_db" mkdir vector_db
if not exist "models" mkdir models

echo.
echo 🔄 Checking .env file...
if not exist ".env" (
    echo GOOGLE_API_KEY=your_google_api_key_here > .env
    echo ✅ Đã tạo .env - Vui lòng chỉnh sửa và thêm API key!
    notepad .env
    pause
)

echo.
echo 🐳 Building Docker image...
docker-compose build --no-cache
if %errorlevel% neq 0 (
    echo ❌ Build thất bại! Kiểm tra lỗi ở trên.
    pause
    goto menu
)
echo ✅ Build hoàn tất!
pause
goto menu

:run
echo.
echo 🔍 Kiểm tra container đang chạy...
docker ps -a | findstr %CONTAINER% >nul
if %errorlevel%==0 (
    echo ⚠️ Container đã tồn tại. Đang xóa...
    docker stop %CONTAINER% >nul 2>&1
    docker rm %CONTAINER% >nul 2>&1
)

echo 🚀 Đang khởi động container...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ❌ Không thể khởi động container!
    pause
    goto menu
)

echo ⏳ Đợi container khởi động (15 giây)...
timeout /t 15 >nul

echo 🔍 Kiểm tra health...
docker exec %CONTAINER% curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Container chưa sẵn sàng, xem log để kiểm tra
) else (
    echo ✅ Container đã sẵn sàng!
    echo 🌐 API: http://localhost:8000
    echo 📖 Docs: http://localhost:8000/docs
)
pause
goto menu

:update
echo.
echo 🧠 Đang cập nhật database...
echo 🗑️ Xoá database cũ...
docker exec %CONTAINER% sh -c "rm -rf /app/vector_db/*"

echo 📊 Setup database mới...
docker exec %CONTAINER% python db_setup.py

echo 📥 Kiểm tra dữ liệu trong /app/data...
docker exec %CONTAINER% ls -1 /app/data

echo 🔄 Ingest documents...
docker exec %CONTAINER% python ingest.py --folder /app/data --force

echo 🔍 Kiểm tra health...
docker exec %CONTAINER% curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Container chưa sẵn sàng, xem log để kiểm tra
) else (
    echo ✅ Container đã sẵn sàng!
    echo 🌐 API: http://localhost:8000
    echo 📖 Docs: http://localhost:8000/docs
)
pause
goto menu

:prompt
echo.
echo ============================================
echo 💬 Chat với Chatbot
echo ============================================
echo Gõ 'menu' để quay lại, 'clear' để xóa màn hình
echo.

:ask
set "user_input="
set /p "user_input=👤 Bạn: "
if /i "%user_input%"=="menu" goto menu
if /i "%user_input%"=="exit" goto menu
if /i "%user_input%"=="clear" (
    cls
    echo 💬 Chat với Chatbot
    echo.
    goto ask
)
if "%user_input%"=="" goto ask

echo 🤖 Bot: 
for /f "delims=" %%A in ('curl -s -X POST http://localhost:8000/chat ^
    -H "Content-Type: application/json; charset=utf-8" ^
    -d "{\"user_id\":\"user123\",\"message\":\"%user_input%\"}"') do (
    set "api_response=%%A"
)

where jq >nul 2>nul
if %errorlevel%==0 (
    echo !api_response! | jq -r ".reply"
) else (
    echo !api_response!
)

echo.
goto ask

:logs
echo.
echo 📜 Hiển thị log container...
echo (Nhấn Ctrl+C để dừng)
echo.
docker logs -f --tail 50 %CONTAINER%
pause
goto menu

:status
echo.
echo 🔍 Kiểm tra trạng thái...
echo.

echo 📦 Container Status:
docker ps -a | findstr %CONTAINER%
if %errorlevel% neq 0 (
    echo ❌ Container không tồn tại
    pause
    goto menu
)

echo.
echo 📊 Container Stats:
docker stats %CONTAINER% --no-stream

echo.
echo 🌐 API Health Check:
docker exec %CONTAINER% curl -s http://localhost:8000/health
if %errorlevel% neq 0 (
    echo ❌ API không phản hồi
)

echo.
echo 📈 Database Info:
docker exec %CONTAINER% curl -s http://localhost:8000/stats

echo.
echo 🖼️ Docker Images:
docker images | findstr chatbot

echo.
echo 💾 Volumes:
docker volume ls | findstr chatbot

pause
goto menu

:stop
echo.
echo 🛑 Đang dừng và xóa container...
docker-compose down
if %errorlevel% neq 0 (
    docker stop %CONTAINER% >nul 2>&1
    docker rm %CONTAINER% >nul 2>&1
)
echo ✅ Container đã được dừng và xóa.
pause
goto menu

:remove_image
echo.
echo 🗑️ Đang xóa Docker image...
docker-compose down 2>nul
docker rmi %IMAGE% 2>nul
docker rmi chatbot-chatbot:latest 2>nul
if %errorlevel% neq 0 (
    echo ⚠️ Không tìm thấy image hoặc lỗi khi xóa
) else (
    echo ✅ Image đã được xóa!
)
pause
goto menu

:clean_all
echo.
echo ⚠️ CẢNH BÁO: Thao tác này sẽ xóa:
echo    - Container
echo    - Image
echo    - Volumes (bao gồm cả database)
echo.
set /p confirm=❓ Bạn có chắc chắn? (yes/no): 
if /i not "%confirm%"=="yes" (
    echo ℹ️ Đã hủy thao tác
    pause
    goto menu
)

echo.
echo 🧹 Cleaning...
docker-compose down -v
docker rmi %IMAGE% 2>nul
docker rmi chatbot-chatbot:latest 2>nul
docker volume prune -f

echo ✅ Đã xóa toàn bộ!
pause
goto menu

:rebuild
echo.
echo 🔄 Rebuild và restart...
docker-compose down
docker-compose build --no-cache
if %errorlevel% neq 0 (
    echo ❌ Build thất bại!
    pause
    goto menu
)
docker-compose up -d
timeout /t 15 >nul
echo ✅ Rebuild hoàn tất!
echo 🌐 API: http://localhost:8000
pause
goto menu

:end
echo.
echo 👋 Tạm biệt! Cảm ơn bạn đã sử dụng Vietnamese Chatbot!
timeout /t 2 >nul
exit
