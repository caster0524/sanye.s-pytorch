'use client';

import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import UploadZone from '@/components/UploadZone';
import ModelSelector from '@/components/ModelSelector';
import StatusDisplay from '@/components/StatusDisplay';
import { detectObjects } from '@/lib/api';
import { Radar, Play, Info } from 'lucide-react';
import type { DetectionResult } from '@/types';

const detectionModels = {
  fasterrcnn_resnet50: {
    name: 'Faster R-CNN',
    description: 'Region-based CNN for accurate object detection (59.9% mAP@50)',
    input_size: [800, 800] as [number, number],
    map50: '59.9%',
  },
};

export default function DetectionPage() {
  const [selectedModel, setSelectedModel] = useState('fasterrcnn_resnet50');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDetect = async () => {
    if (!selectedFile) return;

    setStatus('loading');
    setError(null);
    setResult(null);

    try {
      const response = await detectObjects(selectedFile, selectedModel, confidenceThreshold);
      
      if (!response.ok) {
        throw new Error(`Detection failed: ${response.statusText}`);
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
              <Radar className="w-8 h-8 text-purple-400" />
              <h1 className="text-3xl font-bold text-white">目标检测</h1>
            </div>
            <p className="text-slate-400">
              Detect and localize objects in images using state-of-the-art detection models.
              See bounding boxes with class labels and confidence scores.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Input Section */}
            <div className="space-y-6">
              {/* Model Selection */}
              <div className="card p-6">
                <h2 className="text-lg font-semibold text-white mb-4">Model Selection</h2>
                <ModelSelector
                  models={detectionModels}
                  selectedModel={selectedModel}
                  onModelChange={setSelectedModel}
                  label="Detection Model"
                />
                
                <div className="mt-4 space-y-2">
                  <div className="flex justify-between">
                    <label className="text-sm font-medium text-slate-300">
                      Confidence Threshold
                    </label>
                    <span className="text-sm text-slate-400">{(confidenceThreshold * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={confidenceThreshold * 100}
                    onChange={(e) => setConfidenceThreshold(Number(e.target.value) / 100)}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
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
                onClick={handleDetect}
                disabled={!selectedFile || status === 'loading'}
                className="btn-primary w-full py-3 flex items-center justify-center gap-2"
              >
                {status === 'loading' ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    检测中...
                  </>
                ) : (
                  <>
                    <Play size={20} />
                    Detect Objects
                  </>
                )}
              </button>

              {/* Status */}
              {status !== 'idle' && (
                <StatusDisplay
                  status={status}
                  message={
                    status === 'loading' ? '检测中...jects...' :
                    status === 'success' ? `Found ${result?.count || 0} objects!` :
                    `Error: ${error}`
                  }
                />
              )}
            </div>

            {/* Results Section */}
            <div className="space-y-6">
              {result ? (
                <>
                  {/* Annotated Image */}
                  <div className="card p-6">
                    <h2 className="text-lg font-semibold text-white mb-4">
                      检测结果
                    </h2>
                    {result.annotated_image && (
                      <img
                        src={result.annotated_image}
                        alt="Detection results"
                        className="w-full rounded-lg"
                      />
                    )}
                  </div>

                  {/* Detections List */}
                  <div className="card p-6">
                    <h2 className="text-lg font-semibold text-white mb-4">
                      检测到的物体 ({result.count})
                    </h2>
                    <div className="max-h-64 overflow-y-auto space-y-2">
                      {result.detections.slice(0, 20).map((detection, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg"
                        >
                          <div className="flex items-center gap-3">
                            <span className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-sm font-medium">
                              {detection.class_name}
                            </span>
                          </div>
                          <div className="text-right">
                            <span className="text-purple-400 font-mono">
                              {(detection.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                      {result.detections.length > 20 && (
                        <p className="text-center text-slate-500 text-sm">
                          And {result.detections.length - 20} more...
                        </p>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <div className="card p-12 flex flex-col items-center justify-center text-center">
                  <Radar className="w-16 h-16 text-slate-600 mb-4" />
                  <p className="text-slate-400">
                    Upload an image and click Detect to find objects
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
