# PyTorch 深度学习工作室

![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?style=flat-square&logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=next.js)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

一个全面的 PyTorch 深度学习 Web 平台，支持图像分类、目标检测、图像分割、自然语言处理和 AI 图像生成。

## 功能特性

### 图像分类
- **模型**: ResNet50, VGG16, EfficientNet-B0
- **功能**: Top-5 预测及置信度、可视化图表
- **预训练来源**: torchvision.models (Apache 2.0)

### 目标检测
- **模型**: Faster R-CNN, YOLOv5 (可用时)
- **功能**: 带标签和置信度的边界框
- **预训练来源**: torchvision.models.detection (Apache 2.0)

### 图像分割
- **模型**: DeepLabV3, FCN (全卷积网络)
- **功能**: 语义分割与叠加可视化
- **预训练来源**: torchvision.models.segmentation (Apache 2.0)

### 自然语言处理
- **模型**: BERT 情感分析, GPT-2 文本生成
- **功能**: 情感分析、文本生成、摘要
- **预训练来源**: Hugging Face Transformers (Apache 2.0)

### AI 图像生成
- **模型**: Stable Diffusion 2.1
- **功能**: 文本到图像生成，支持自定义参数
- **预训练来源**: Stability AI (CreativeML Open RAIL-M)

### 模型训练
- 基于用户数据集的自定义模型训练
- 实时进度监控
- 检查点保存

### 模型管理
- 上传自定义 PyTorch (.pt, .pth) 或 ONNX (.onnx) 模型
- 模型验证和元数据存储

## 预训练模型及来源

本项目使用以下预训练模型：

| 模型 | 任务 | 来源 | 许可证 |
|-------|------|--------|---------|
| ResNet50 | 分类 | torchvision.models | Apache 2.0 |
| VGG16 | 分类 | torchvision.models | Apache 2.0 |
| EfficientNet-B0 | 分类 | torchvision.models | Apache 2.0 |
| Faster R-CNN | 检测 | torchvision.models.detection | Apache 2.0 |
| DeepLabV3 | 分割 | torchvision.models.segmentation | Apache 2.0 |
| FCN ResNet50 | 分割 | torchvision.models.segmentation | Apache 2.0 |
| BERT | 自然语言处理 | Hugging Face | Apache 2.0 |
| GPT-2 | 文本生成 | OpenAI | Modified MIT |
| Stable Diffusion 2.1 | 图像生成 | Stability AI | CreativeML Open RAIL-M |
| sentence-transformers | 向量嵌入 | Hugging Face | Apache 2.0 |

### 模型许可说明

- **torchvision 模型**: Apache 2.0 许可证 - 允许商业使用
- **BERT (Hugging Face)**: Apache 2.0 许可证 - 允许商业使用
- **GPT-2**: Modified MIT 许可证 - 允许商业使用
- **Stable Diffusion**: CreativeML Open RAIL-M 许可证 - 允许在限制条件下商业使用

详细许可证信息请参阅:
- [PyTorch torchvision 许可证](https://github.com/pytorch/vision/blob/main/LICENSE)
- [Hugging Face Transformers 许可证](https://github.com/huggingface/transformers/blob/main/LICENSE)
- [GPT-2 许可证](https://github.com/openai/gpt-2/blob/master/LICENSE)
- [Stable Diffusion 许可证](https://github.com/Stability-AI/generative-models/blob/main/LICENSE)

## 安装

### 环境要求
- Python 3.10+
- Node.js 18+
- PyTorch (如需 GPU 支持，请安装 CUDA 版本)
- npm 或 pnpm

### 后端安装

```bash
cd backend
pip install -r requirements.txt
python main.py
```

### 前端安装

```bash
cd frontend
pnpm install
pnpm dev
```

### 环境变量（可选）

```bash
# 后端
HOST=0.0.0.0
PORT=8000

# 前端
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 使用方法

1. 启动后端服务: `python backend/main.py`
2. 启动前端: `pnpm dev` (在 frontend 目录下)
3. 在浏览器中打开 http://localhost:5000

## 项目结构

```
pytorch-web-app/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── routes/              # API 端点
│   │   ├── classify.py       # 图像分类
│   │   ├── detect.py         # 目标检测
│   │   ├── segment.py        # 图像分割
│   │   ├── nlp.py           # 自然语言处理
│   │   ├── generate.py      # AIGC 生成
│   │   ├── models.py        # 模型管理
│   │   └── train.py         # 训练接口
│   ├── services/             # 业务逻辑
│   └── utils/                # 工具函数
├── frontend/                 # Next.js 应用
│   ├── src/
│   │   ├── app/              # 页面
│   │   ├── components/       # UI 组件
│   │   ├── lib/             # API 客户端
│   │   └── types/           # TypeScript 类型
│   └── public/               # 静态文件
├── public/                   # 上传文件
├── DESIGN.md                 # 设计规范
├── AGENTS.md                 # 开发指南
└── README.md                 # 本文件
```

## API 接口

### 文件上传
- `POST /api/upload` - 上传文件
- `POST /api/upload/batch` - 批量上传

### 图像分类
- `POST /api/classify` - 分类图像
- `GET /api/classify/models` - 获取模型列表

### 目标检测
- `POST /api/detect` - 检测物体
- `GET /api/detect/models` - 获取模型列表

### 图像分割
- `POST /api/segment` - 分割图像
- `GET /api/segment/models` - 获取模型列表

### 自然语言处理
- `POST /api/nlp/sentiment` - 情感分析
- `POST /api/nlp/generate` - 文本生成
- `POST /api/nlp/summarize` - 文本摘要

### 图像生成
- `POST /api/generate` - 根据提示词生成图像

### 模型管理
- `GET /api/models` - 获取所有模型
- `POST /api/models/upload` - 上传自定义模型

### 模型训练
- `POST /api/train` - 开始训练
- `GET /api/train/status/{job_id}` - 获取状态
- `GET /api/train/jobs` - 获取任务列表

## 部署

### 后端部署 (GPU)
```bash
cd backend
pip install -r requirements.txt
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

### 前端部署
```bash
cd frontend
pnpm build
pnpm start
```

## 贡献

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [torchvision](https://pytorch.org/vision/) - 计算机视觉模型
- [Hugging Face](https://huggingface.co/) - NLP 模型
- [Stable Diffusion](https://stability.ai/) - 图像生成模型
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Python Web 框架
- [Next.js](https://nextjs.org/) - React 框架
