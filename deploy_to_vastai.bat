@echo off
echo 🚀 Connecting to Vast.ai instance and deploying chest X-ray AI system...
echo.

REM Instance details
set INSTANCE_IP=85.10.218.46
set INSTANCE_PORT=46499
set SSH_KEY=vastai_key

REM Check if SSH key exists
if not exist "%SSH_KEY%" (
    echo ❌ SSH key '%SSH_KEY%' not found!
    echo Please make sure you're in the correct directory and the SSH key exists.
    echo Also, make sure you've added the public key to your Vast.ai account.
    pause
    exit /b 1
)

echo 📡 Connecting to %INSTANCE_IP%:%INSTANCE_PORT%...
echo.

REM Upload and run deployment script
ssh -i %SSH_KEY% -o StrictHostKeyChecking=no root@%INSTANCE_IP% -p %INSTANCE_PORT% "curl -s -L https://raw.githubusercontent.com/MAbdullahTrq/chest-xray-ai-poc/master/deploy_to_vastai.sh -o deploy.sh && chmod +x deploy.sh && ./deploy.sh"

echo.
echo ✅ Deployment completed!
echo.
echo 🌐 Your chest X-ray AI system should now be accessible at:
echo    Frontend: http://%INSTANCE_IP%:3000
echo    API:      http://%INSTANCE_IP%:8000
echo    API Docs: http://%INSTANCE_IP%:8000/docs
echo.
echo 🔗 To connect manually: ssh -i %SSH_KEY% root@%INSTANCE_IP% -p %INSTANCE_PORT%
echo.
pause
