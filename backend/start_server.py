"""延迟加载推理引擎的后端启动脚本"""
import uvicorn
import sys
from pathlib import Path

# 确保 backend 在路径中
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("Starting PyTorch Deep Learning Server...")
    print("Device: GPU (RTX 4060)")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )