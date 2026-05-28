'use client';

import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import UploadZone from '@/components/UploadZone';
import ModelSelector from '@/components/ModelSelector';
import StatusDisplay from '@/components/StatusDisplay';
import { segmentImage } from '@/lib/api';
import { Layers, Play } from 'lucide-react';
import type { SegmentationResult } from '@/types';

const segmentationModels = {
  deeplabv3_resnet50: {
    name: 'DeepLabV3',
    description: 'DeepLab semantic image segmentation (77.7% mIoU)',
    input_size: [520, 520] as [number, number],
    mIoU: '77.7%',
  },
  fcn_resnet50: {
    name: 'FCN ResNet50',
    description: 'Fully Convolutional Network for segmentation (72.6% mIoU)',
    input_size: [520, 520] as [number, number],
    mIoU: '72.6%',
  },
};

export default function SegmentationPage() {
  const [selectedModel, setSelectedModel] = useState('deeplabv3_resnet50');
  const [overlayAlpha, setOverlayAlpha] = useState(0.5);
  const [viewMode, setViewMode] = useState<'overlay' | 'mask'>('overlay');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSegment = async () => {
    if (!selectedFile) return;

    setStatus('loading');
    setError(null);
    setResult(null);

    try {
      const response = await segmentImage(selectedFile, selectedModel, overlayAlpha);
      
      if (!response.ok) {
        throw new Error(`Segmentation failed: ${response.statusText}`);
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
              <Layers className="w-8 h-8 text-green-400" />
              <h1 className="text-3xl font-bold text-white">图像分割</h1>
            </div>
            <p className="text-slate-400">
              Perform semantic segmentation on images. Each pixel is classified into a category,
              creating a detailed mask showing different regions.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Input Section */}
            <div className="space-y-6">
              {/* Model Selection */}
              <div className="card p-6">
                <h2 className="text-lg font-semibold text-white mb-4">Model Selection</h2>
                <ModelSelector
                  models={segmentationModels}
                  selectedModel={selectedModel}
                  onModelChange={setSelectedModel}
                  label="Segmentation Model"
                />
                
                <div className="mt-4 space-y-2">
                  <div className="flex justify-between">
                    <label className="text-sm font-medium text-slate-300">
                      Overlay Alpha
                    </label>
                    <span className="text-sm text-slate-400">{(overlayAlpha * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={overlayAlpha * 100}
                    onChange={(e) => setOverlayAlpha(Number(e.target.value) / 100)}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-green-500"
                  />
                </div>
              </div>

              {/* Upload */}
              <div className="card p-6">
                <h2 className="text-lg font-semibold text-white mb-4">Upload Image</h2>
                <UploadZone
                  onFileSelect={setSelectedFile}
                  accept="image/*"
                  maxSize={10 * 1024 * 1024}
                  disabled={status === 'loading'}
                />
              </div>

              {/* Actions */}
              <button
                onClick={handleSegment}
                disabled={!selectedFile || status === 'loading'}
                className="btn-primary w-full py-3 flex items-center justify-center gap-2"
              >
                {status === 'loading' ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    分割中...
                  </>
                ) : (
                  <>
                    <Play size={20} />
                    Segment Image
                  </>
                )}
              </button>

              {/* Status */}
              {status !== 'idle' && (
                <StatusDisplay
                  status={status}
                  message={
                    status === 'loading' ? 'Processing segmentation...' :
                    status === 'success' ? 'Segmentation complete!' :
                    `Error: ${error}`
                  }
                />
              )}
            </div>

            {/* Results Section */}
            <div className="space-y-6">
              {result ? (
                <>
                  {/* Segmentation Result */}
                  <div className="card p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-lg font-semibold text-white">Segmentation Result</h2>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setViewMode('overlay')}
                          className={`px-3 py-1 text-sm rounded ${
                            viewMode === 'overlay' 
                              ? 'bg-green-500/20 text-green-400' 
                              : 'text-slate-400 hover:text-white'
                          }`}
                        >
                          Overlay
                        </button>
                        <button
                          onClick={() => setViewMode('mask')}
                          className={`px-3 py-1 text-sm rounded ${
                            viewMode === 'mask' 
                              ? 'bg-green-500/20 text-green-400' 
                              : 'text-slate-400 hover:text-white'
                          }`}
                        >
                          Mask Only
                        </button>
                      </div>
                    </div>
                    <img
                      src={viewMode === 'overlay' ? result.overlay : result.mask}
                      alt="Segmentation result"
                      className="w-full rounded-lg"
                    />
                  </div>

                  {/* Class Distribution */}
                  <div className="card p-6">
                    <h2 className="text-lg font-semibold text-white mb-4">
                      Class Distribution ({result.num_classes} classes)
                    </h2>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      {Object.entries(result.class_distribution).slice(0, 10).map(([cls, count]) => (
                        <div key={cls} className="flex justify-between p-2 bg-slate-800/50 rounded">
                          <span className="text-slate-400">Class {cls}</span>
                          <span className="text-green-400 font-mono">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="card p-12 flex flex-col items-center justify-center text-center">
                  <Layers className="w-16 h-16 text-slate-600 mb-4" />
                  <p className="text-slate-400">
                    上传图像并点击分割按钮以查看结果
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
