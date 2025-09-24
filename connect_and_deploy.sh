#!/bin/bash

# Connect to Vast.ai instance and deploy chest X-ray AI system
# Usage: ./connect_and_deploy.sh

set -e

echo "🚀 Connecting to Vast.ai instance and deploying chest X-ray AI system..."

# Instance details
INSTANCE_IP="85.10.218.46"
INSTANCE_PORT="46499"
SSH_KEY="vastai_key"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key '$SSH_KEY' not found!"
    echo "Please make sure you're in the correct directory and the SSH key exists."
    exit 1
fi

echo "📡 Connecting to $INSTANCE_IP:$INSTANCE_PORT..."

# Copy deployment script to instance and run it
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no root@"$INSTANCE_IP" -p "$INSTANCE_PORT" << 'ENDSSH'

# Download and run deployment script
echo "📥 Downloading deployment script..."
curl -s -o deploy_chest_xray_ai.sh https://raw.githubusercontent.com/MAbdullahTrq/chest-xray-ai-poc/master/deploy_to_vastai.sh

# Make it executable
chmod +x deploy_chest_xray_ai.sh

# Run deployment
echo "🚀 Starting deployment..."
./deploy_chest_xray_ai.sh

ENDSSH

echo ""
echo "✅ Deployment completed!"
echo ""
echo "🌐 Your chest X-ray AI system should now be accessible at:"
echo "   Frontend: http://$INSTANCE_IP:3000"
echo "   API:      http://$INSTANCE_IP:8000"
echo "   API Docs: http://$INSTANCE_IP:8000/docs"
echo ""
echo "🔗 To connect manually: ssh -i $SSH_KEY root@$INSTANCE_IP -p $INSTANCE_PORT"
