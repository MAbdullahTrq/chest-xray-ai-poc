# TensorDock Setup Guide for Chest X-ray AI POC

This guide walks you through setting up TensorDock for your chest X-ray AI diagnostic tool.

## 🚀 Quick Start Summary

1. **Sign up** for TensorDock account
2. **Deploy** RTX 3090 instance 
3. **Install** dependencies
4. **Run** the POC application
5. **Access** via public IP

**Estimated setup time**: 15-20 minutes  
**Cost**: ~$0.29/hour for RTX 3090

---

## 📋 Prerequisites

- Valid email address for TensorDock account
- Credit card for billing (minimum $10 deposit)
- Basic familiarity with command line/terminal
- Your POC code ready to deploy

---

## 1. TensorDock Account Setup

### Step 1.1: Create Account
1. Go to [TensorDock.com](https://www.tensordock.com)
2. Click **"Sign Up"** 
3. Fill in your details:
   - Email address
   - Strong password
   - Agree to terms of service
4. **Verify your email** (check spam folder if needed)

### Step 1.2: Add Payment Method
1. Log in to your TensorDock dashboard
2. Navigate to **"Billing"** section
3. Click **"Add Payment Method"**
4. Enter credit card details
5. **Add initial credit** (minimum $10 recommended)

### Step 1.3: Understand Pricing
```
RTX 3090 Instance Pricing:
- Hourly rate: $0.29/hour
- Daily cost: ~$7/day (24/7)
- Weekly cost: ~$49/week
- Monthly cost: ~$212/month

Storage:
- 50GB SSD: $0.10/GB/month = $5/month
- 100GB SSD: $10/month (recommended)
```

---

## 2. Deploy RTX 3090 Instance

### Step 2.1: Create New Instance
1. In TensorDock dashboard, click **"Deploy"**
2. **Select GPU**: Choose **"RTX 3090"** (24GB VRAM)
3. **Select Region**: Choose closest region for better latency
4. **Configure Instance**:
   ```
   GPU: RTX 3090 (24GB)
   CPU: 8 vCPUs (included)
   RAM: 30GB (included)
   Storage: 100GB SSD ($10/month)
   OS: Ubuntu 22.04 LTS
   ```

### Step 2.2: Configure Access
1. **SSH Key Setup** (Recommended):
   - Generate SSH key if you don't have one:
     ```bash
     ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
     ```
   - Copy public key content:
     ```bash
     cat ~/.ssh/id_rsa.pub
     ```
   - Paste in TensorDock SSH key field

2. **Or use Password** (Less secure):
   - Check "Enable password authentication"
   - Set a strong password

### Step 2.3: Deploy Instance
1. **Review configuration**:
   ```
   Instance Type: RTX 3090
   Hourly Cost: $0.29
   Monthly Storage: $10
   OS: Ubuntu 22.04
   ```
2. Click **"Deploy Instance"**
3. **Wait 2-5 minutes** for deployment
4. **Note down**:
   - Public IP address
   - SSH connection details
   - Instance ID

---

## 3. Connect to Your Instance

### Step 3.1: SSH Connection
```bash
# Using SSH key (recommended)
ssh root@YOUR_INSTANCE_IP

# Using password
ssh root@YOUR_INSTANCE_IP
# Enter password when prompted
```

### Step 3.2: Verify GPU Access
```bash
# Check GPU status
nvidia-smi

# Expected output should show:
# - RTX 3090 with 24GB memory
# - CUDA version
# - Driver version
```

### Step 3.3: Update System
```bash
# Update package list
apt update && apt upgrade -y

# Install essential tools
apt install -y wget curl git vim htop tree
```

---

## 4. Install Dependencies

### Step 4.1: Install Python and Pip
```bash
# Python should be pre-installed, verify
python3 --version
pip3 --version

# If pip is missing
apt install -y python3-pip
```

### Step 4.2: Install PyTorch with CUDA
```bash
# Install PyTorch with CUDA 11.8 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch GPU access
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Step 4.3: Install Additional Dependencies
```bash
# Install system dependencies for medical imaging
apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Install Python medical imaging libraries
pip3 install opencv-python pillow pydicom SimpleITK scikit-image
```

---

## 5. Deploy Your POC Application

### Step 5.1: Upload Your Code
```bash
# Option 1: Git clone (if you have a repository)
git clone https://github.com/your-username/chest_xray_poc.git
cd chest_xray_poc

# Option 2: Upload via SCP from local machine
# Run this from your local machine:
scp -r chest_xray_poc/ root@YOUR_INSTANCE_IP:/root/
```

### Step 5.2: Install Project Dependencies
```bash
cd chest_xray_poc

# Install requirements
pip3 install -r requirements.txt

# This will install:
# - FastAPI and Uvicorn
# - TorchXRayVision
# - All other dependencies
```

### Step 5.3: Download AI Models
```bash
# TorchXRayVision will download models automatically on first run
# Pre-download to save time during first analysis:
python3 -c "
import torchxrayvision as xrv
print('Downloading models...')
model = xrv.models.get_model('densenet121-res224-all')
print('Models ready!')
"
```

---

## 6. Run the Application

### Step 6.1: Start the Backend API
```bash
# Navigate to project directory
cd /root/chest_xray_poc

# Start the FastAPI server
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# You should see:
# INFO: Uvicorn running on http://0.0.0.0:8000
# INFO: Loading chest X-ray model...
# INFO: Model loaded successfully!
```

### Step 6.2: Test API Connection
```bash
# In a new terminal (or use screen/tmux)
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","message":"System status check","model_loaded":true,"gpu_available":true,"gpu_count":1}
```

### Step 6.3: Serve Frontend (Simple Method)
```bash
# In another terminal/screen session
cd /root/chest_xray_poc/frontend
python3 -m http.server 3000

# Frontend will be available at: http://YOUR_INSTANCE_IP:3000
```

---

## 7. Access Your Application

### Step 7.1: Configure Firewall (if needed)
```bash
# Allow HTTP traffic (ports 3000 and 8000)
ufw allow 3000
ufw allow 8000
ufw enable
```

### Step 7.2: Access URLs
- **Frontend**: `http://YOUR_INSTANCE_IP:3000`
- **API Docs**: `http://YOUR_INSTANCE_IP:8000/docs`
- **Health Check**: `http://YOUR_INSTANCE_IP:8000/health`

### Step 7.3: Test the Application
1. **Open frontend** in your browser
2. **Upload a chest X-ray** image
3. **Click "Analyze X-ray"**
4. **Review results** (should process in 2-3 seconds)

---

## 8. Production Optimizations

### Step 8.1: Use Process Manager
```bash
# Install PM2 for process management
npm install -g pm2

# Create ecosystem file
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    {
      name: 'xray-api',
      script: 'python3',
      args: ['-m', 'uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8000'],
      cwd: '/root/chest_xray_poc',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
    },
    {
      name: 'xray-frontend',
      script: 'python3',
      args: ['-m', 'http.server', '3000'],
      cwd: '/root/chest_xray_poc/frontend',
      instances: 1,
      autorestart: true,
      watch: false,
    }
  ]
};
EOF

# Start services
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

### Step 8.2: Setup Nginx (Optional)
```bash
# Install Nginx
apt install -y nginx

# Configure reverse proxy
cat > /etc/nginx/sites-available/xray-app << 'EOF'
server {
    listen 80;
    server_name YOUR_INSTANCE_IP;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

# Enable site
ln -s /etc/nginx/sites-available/xray-app /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

---

## 9. Monitoring and Management

### Step 9.1: Monitor GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or install htop for system monitoring
htop
```

### Step 9.2: Check Application Logs
```bash
# PM2 logs
pm2 logs

# Or if running manually
tail -f /var/log/nginx/access.log
```

### Step 9.3: Cost Management
```bash
# Check current usage in TensorDock dashboard
# Stop instance when not in use:
# Dashboard > Instances > Stop

# Estimate costs:
# Current session: Hours used × $0.29
# Storage: Always charged ($10/month for 100GB)
```

---

## 10. Troubleshooting

### Common Issues and Solutions

#### Issue: CUDA Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Solution: Restart the application
pm2 restart xray-api
```

#### Issue: Model Loading Failed
```bash
# Check internet connection
ping google.com

# Manually download models
python3 -c "import torchxrayvision as xrv; xrv.models.get_model('densenet121-res224-all')"
```

#### Issue: Frontend Can't Connect to API
```bash
# Check if API is running
curl http://localhost:8000/health

# Check firewall
ufw status

# Restart services
pm2 restart all
```

#### Issue: Slow Processing
```bash
# Verify GPU is being used
nvidia-smi

# Check system resources
htop

# Ensure PyTorch is using GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 11. Security Best Practices

### Step 11.1: Secure SSH
```bash
# Disable password authentication (if using SSH keys)
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart ssh
```

### Step 11.2: Setup Firewall
```bash
# Configure UFW
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80
ufw allow 3000
ufw allow 8000
ufw enable
```

### Step 11.3: Regular Updates
```bash
# Create update script
cat > /root/update.sh << 'EOF'
#!/bin/bash
apt update && apt upgrade -y
pip3 install --upgrade pip
pip3 install --upgrade -r /root/chest_xray_poc/requirements.txt
EOF

chmod +x /root/update.sh

# Run weekly updates
crontab -e
# Add: 0 2 * * 0 /root/update.sh
```

---

## 12. Backup and Recovery

### Step 12.1: Backup Your Work
```bash
# Create backup script
cat > /root/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf /root/xray_poc_backup_$DATE.tar.gz /root/chest_xray_poc
echo "Backup created: xray_poc_backup_$DATE.tar.gz"
EOF

chmod +x /root/backup.sh
./backup.sh
```

### Step 12.2: Download Backups
```bash
# From your local machine
scp root@YOUR_INSTANCE_IP:/root/xray_poc_backup_*.tar.gz ./
```

---

## 13. Scaling and Advanced Features

### Step 13.1: Load Balancing (Multiple Instances)
```bash
# Deploy multiple instances
# Use nginx for load balancing
# Configure health checks
```

### Step 13.2: Auto-scaling
```bash
# Monitor CPU/GPU usage
# Scale up/down based on demand
# Use TensorDock API for automation
```

---

## 📊 Cost Summary

### Expected Costs for Different Usage Patterns:

| Usage Pattern | Hours/Month | GPU Cost | Storage | Total |
|---------------|-------------|----------|---------|-------|
| **Development** (40h) | 40 | $11.60 | $10 | $21.60 |
| **Testing** (100h) | 100 | $29.00 | $10 | $39.00 |
| **Demo/Production** (200h) | 200 | $58.00 | $10 | $68.00 |
| **24/7 Production** (730h) | 730 | $211.70 | $10 | $221.70 |

### Cost Optimization Tips:
- **Stop instances** when not in use
- **Use spot instances** for development (if available)
- **Monitor usage** regularly in dashboard
- **Set billing alerts** to avoid surprises

---

## ✅ Success Checklist

After completing this guide, you should have:

- [ ] TensorDock account with RTX 3090 instance
- [ ] SSH access to your instance
- [ ] PyTorch with CUDA working
- [ ] TorchXRayVision models downloaded
- [ ] FastAPI backend running on port 8000
- [ ] Frontend accessible via web browser
- [ ] Successful X-ray analysis (2-3 seconds)
- [ ] Process management with PM2
- [ ] Basic security configurations
- [ ] Backup procedures in place

---

## 🆘 Support and Resources

### TensorDock Support:
- **Documentation**: [TensorDock Docs](https://docs.tensordock.com)
- **Discord**: TensorDock Community Server
- **Email**: support@tensordock.com

### Technical Support:
- **PyTorch Issues**: [PyTorch Forums](https://discuss.pytorch.org/)
- **TorchXRayVision**: [GitHub Issues](https://github.com/mlmed/torchxrayvision/issues)
- **FastAPI**: [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Emergency Contacts:
- **Instance Issues**: TensorDock Support
- **Billing Questions**: billing@tensordock.com
- **Technical Problems**: Check logs first, then community forums

---

**🎉 Congratulations!** You now have a fully functional chest X-ray AI diagnostic tool running on TensorDock. The system should process X-rays in 2-3 seconds and cost approximately $0.0002 per analysis.

Remember to monitor your usage and costs, and stop instances when not needed to optimize expenses.
