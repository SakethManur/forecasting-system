@echo off
cd /d D:\microgcc\forecasting-system
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 1> logs\uvicorn-detached.log 2>&1
