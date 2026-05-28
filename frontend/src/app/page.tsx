'use client';

import { useEffect, useState } from 'react';
import Sidebar from '@/components/Sidebar';
import {
  Image,
  Radar,
  Layers,
  FileText,
  Sparkles,
  Brain,
  Cpu,
  Database,
  ArrowRight,
} from 'lucide-react';
import Link from 'next/link';
import { healthCheck } from '@/lib/api';

const features = [
  {
    icon: Image,
    title: '图像分类',
    description: '使用 ResNet、VGG 和 EfficientNet 模型对图像进行分类',
    href: '/classification',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    icon: Radar,
    title: '目标检测',
    description: '使用 YOLO 和 Faster R-CNN 检测图像中的物体',
    href: '/detection',
    color: 'from-purple-500 to-pink-500',
  },
  {
    icon: Layers,
    title: '图像分割',
    description: '使用 DeepLabV3 和 FCN 进行语义分割',
    href: '/segmentation',
    color: 'from-green-500 to-emerald-500',
  },
  {
    icon: FileText,
    title: '自然语言处理',
    description: '使用 BERT 和 GPT-2 进行情感分析和文本生成',
    href: '/nlp',
    color: 'from-orange-500 to-yellow-500',
  },
  {
    icon: Sparkles,
    title: 'AI 图像生成',
    description: '使用 Stable Diffusion 根据文本提示生成图像',
    href: '/generation',
    color: 'from-pink-500 to-rose-500',
  },
  {
    icon: Brain,
    title: '模型训练',
    description: '在您自己的数据集上训练自定义模型',
    href: '/training',
    color: 'from-indigo-500 to-purple-500',
  },
];

const stats = [
  { label: '预训练模型', value: '12+', icon: Brain },
  { label: '支持任务', value: '5', icon: Cpu },
  { label: '数据集格式', value: '10+', icon: Database },
];

export default function Home() {
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected'>('disconnected');

  useEffect(() => {
    healthCheck().then((res) => {
      if (res.status === 'healthy') {
        setBackendStatus('connected');
      }
    }).catch(() => {
      setBackendStatus('disconnected');
    });
  }, []);

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto p-8">
          {/* Header */}
          <div className="mb-12">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
              <span className="text-sm text-slate-400">
                {backendStatus === 'connected' ? '后端已连接' : '后端离线'}
              </span>
            </div>
            <h1 className="text-4xl font-bold text-white mb-4">
              欢迎使用 <span className="text-gradient">PyTorch 深度学习工作室</span>
            </h1>
            <p className="text-xl text-slate-400 max-w-2xl">
              一个全面的深度学习 Web 平台。图像分类、目标检测、图像分割、
              文本处理、AI 艺术生成 — 一切尽在掌握。
            </p>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-6 mb-12">
            {stats.map((stat) => (
              <div key={stat.label} className="card p-6 flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-blue-500/20 flex items-center justify-center">
                  <stat.icon className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <div className="text-2xl font-bold text-white">{stat.value}</div>
                  <div className="text-sm text-slate-400">{stat.label}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Features Grid */}
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-white mb-6">功能模块</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {features.map((feature) => (
                <Link
                  key={feature.title}
                  href={feature.href}
                  className="card p-6 group hover:scale-[1.02] transition-transform"
                >
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4`}>
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                    {feature.title}
                    <ArrowRight className="w-4 h-4 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                  </h3>
                  <p className="text-sm text-slate-400">{feature.description}</p>
                </Link>
              ))}
            </div>
          </div>

          {/* Quick Start */}
          <div className="card p-8">
            <h2 className="text-2xl font-bold text-white mb-4">快速开始</h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-medium text-white mb-3">1. 选择任务</h3>
                <p className="text-slate-400 mb-4">
                  从图像分类、目标检测、图像分割、自然语言处理或 AI 图像生成中选择。
                </p>
              </div>
              <div>
                <h3 className="text-lg font-medium text-white mb-3">2. 上传或输入</h3>
                <p className="text-slate-400 mb-4">
                  上传图像或输入文本。使用预训练模型或上传您自己的模型。
                </p>
              </div>
              <div>
                <h3 className="text-lg font-medium text-white mb-3">3. 处理</h3>
                <p className="text-slate-400 mb-4">
                  让 PyTorch 模型处理您的数据。通过可视化查看结果。
                </p>
              </div>
              <div>
                <h3 className="text-lg font-medium text-white mb-3">4. 导出</h3>
                <p className="text-slate-400">
                  下载带标注的图像、结果或训练好的模型。
                </p>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="mt-12 text-center text-slate-500 text-sm">
            <p>由 PyTorch、FastAPI 和 Next.js 提供支持</p>
            <p className="mt-1">支持 CUDA 加速（当可用时）</p>
          </div>
        </div>
      </main>
    </div>
  );
}
