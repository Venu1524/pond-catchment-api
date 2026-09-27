#!/bin/bash
set -e

echo "=== Managing Village Pond Planning Services ==="

# 1. Kill any existing uvicorn or python http.server on port 3000 and 6000
echo "Stopping old processes..."
pkill -f "uvicorn main:app" || true
pkill -f "http.server 6000" || true
sleep 1

# Check if ports are freed
fuser -k 3000/tcp || true
fuser -k 6000/tcp || true
sleep 1

cd /home/student/pond-catchment-api

# 2. Start Backend on port 3000 (accessible externally as http://10.1.75.79:3289)
echo "Starting Backend API on port 3000..."
nohup ./venv/bin/uvicorn main:app --host 0.0.0.0 --port 3000 > /home/student/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# 3. Start Frontend Web Server on port 6000 (accessible externally as http://10.1.75.79:6289)
echo "Starting Frontend Web Server on port 6000..."
cd /home/student/pond-frontend
nohup python3 -m http.server 6000 --bind 0.0.0.0 > /home/student/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

sleep 2

# Verify both are running
echo "=== Verifying Listening Ports ==="
netstat -tuln | grep -E '3000|6000'

echo "=== Testing Backend Health ==="
curl -s http://127.0.0.1:3000/health || echo "Backend health check failed"
echo ""

echo "=== Testing Frontend HTTP ==="
curl -s -I http://127.0.0.1:6000/ | head -n 5 || echo "Frontend check failed"

echo "=== SUCCESS! Both services are running in background ==="
