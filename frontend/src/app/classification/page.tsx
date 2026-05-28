'use client';

import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import UploadZone from '@/components/UploadZone';
import ModelSelector from '@/components/ModelSelector';
import PredictionChart from '@/components/PredictionChart';
import StatusDisplay from '@/components/StatusDisplay';
import { classifyImage } from '@/lib/api';
import { Image, Play, Info } from 'lucide-react';
import type { ClassificationResult } from '@/types';

const classificationModels = {
  resnet50: {
    name: 'ResNet50',
    description: '深度残差网络用于图像识别 (ImageNet 上 76.1% top-1 准确率)',
    input_size: [224, 224] as [number, number],
    accuracy: '76.1%',
  },
  vgg16: {
    name: 'VGG16',
    description: '视觉几何组网络 (71.1% top-1 准确率)',
    input_size: [224, 224] as [number, number],
    accuracy: '71.1%',
  },
  efficientnet_b0: {
    name: 'EfficientNet-B0',
    description: '高效卷积神经网络 (77.1% top-1 准确率)',
    input_size: [224, 224] as [number, number],
    accuracy: '77.1%',
  },
};

export default function ClassificationPage() {
  const [selectedModel, setSelectedModel] = useState('resnet50');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleClassify = async () => {
    if (!selectedFile) return;

    setStatus('loading');
    setError(null);
    setResult(null);

    try {
      const response = await classifyImage(selectedFile, selectedModel, 5);
      
      if (!response.ok) {
        throw new Error(`Classification failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
      setStatus('success');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setStatus('error');
    }
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto p-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-4">
              <Image className="w-8 h-8 text-blue-400" />
              <h1 className="text-3xl font-bold text-white">图像分类</h1>
            </div>
            <p className="text-slate-400">
              使用最先进的预训练 PyTorch 模型对图像进行分类。
              上传图像并选择模型以查看预测结果。
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Input Section */}
            <div className="space-y-6">
              {/* Model Selection */}
              <div className="card p-6">
                <h2 className="text-lg font-semibold text-white mb-4">选择模型</h2>
                <ModelSelector
                  models={classificationModels}
                  selectedModel={selectedModel}
                  onModelChange={setSelectedModel}
                  label="分类模型"
                />
              </div>

              {/* Upload */}
              <div className="card p-6">
                <h2 className="text-lg font-semibold text-white mb-4">上传图像</h2>
                <UploadZone
                  onFileSelect={setSelectedFile}
                  accept="image/*"
                  maxSize={10 * 1024 * 1024}
                  disabled={status === 'loading'}
                />
              </div>

              {/* Actions */}
              <button
                onClick={handleClassify}
                disabled={!selectedFile || status === 'loading'}
                className="btn-primary w-full py-3 flex items-center justify-center gap-2"
              >
                {status === 'loading' ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    分类中...
                  </>
                ) : (
                  <>
                    <Play size={20} />
                    开始分类
                  </>
                )}
              </button>

              {/* Status */}
              {status !== 'idle' && (
                <StatusDisplay
                  status={status}
                  message={
                    status === 'loading' ? '正在处理图像...' :
                    status === 'success' ? '分类完成！' :
                    `错误: ${error}`
                  }
                />
              )}
            </div>

            {/* Results Section */}
            <div className="space-y-6">
              {result ? (
                <>
                  {/* Predictions Chart */}
                  <div className="card p-6">
                    <h2 className="text-lg font-semibold text-white mb-4">
                      Top 预测结果
                    </h2>
                    <PredictionChart data={result.predictions} maxItems={5} />
                  </div>

                  {/* Predictions List */}
                  <div className="card p-6">
                    <h2 className="text-lg font-semibold text-white mb-4">
                      预测详情
                    </h2>
                    <div className="space-y-3">
                      {result.predictions.map((pred, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg"
                        >
                          <div className="flex items-center gap-3">
                            <span className="w-8 h-8 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center text-sm font-bold">
                              {index + 1}
                            </span>
                            <span className="text-white font-medium">{pred.class}</span>
                          </div>
                          <div className="text-right">
                            <span className="text-blue-400 font-mono">
                              {(pred.probability * 100).toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Model Info */}
                  <div className="card p-6">
                    <div className="flex items-center gap-2 text-slate-400 text-sm">
                      <Info size={16} />
                      <span>模型: {result.model_info.name}</span>
                      <span className="text-slate-600">|</span>
                      <span>准确率: {result.model_info.accuracy}</span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="card p-12 flex flex-col items-center justify-center text-center">
                  <Image className="w-16 h-16 text-slate-600 mb-4" />
                  <p className="text-slate-400">
                    上传图像并点击分类按钮以查看结果
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
