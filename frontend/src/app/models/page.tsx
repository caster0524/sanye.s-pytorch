'use client';

import { useState, useEffect } from 'react';
import Sidebar from '@/components/Sidebar';
import UploadZone from '@/components/UploadZone';
import StatusDisplay from '@/components/StatusDisplay';
import { getModels, uploadModel, validateModel } from '@/lib/api';
import { Upload, Database, Brain, Trash2, Info } from 'lucide-react';

interface CustomModel {
  name: string;
  filename: string;
  path: string;
  format: string;
  size: number;
  uploaded: string;
  metadata: Record<string, unknown>;
}

interface ModelsData {
  pretrained: {
    classification: Record<string, { name: string; description: string; accuracy?: string }>;
    detection: Record<string, { name: string; description: string; map50?: string }>;
    segmentation: Record<string, { name: string; description: string; mIoU?: string }>;
    nlp: Record<string, { name: string; description: string; type?: string }>;
  };
  custom: CustomModel[];
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelsData | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState('classification');

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const response = await getModels();
      if (response.ok) {
        const data = await response.json();
        // Transform flat models array from backend into pretrained/custom structure
        const modelList = data.models || [];
        const pretrained: ModelsData['pretrained'] = {
          classification: {},
          detection: {},
          segmentation: {},
          nlp: {},
        };
        const custom: ModelsData['custom'] = [];
        for (const m of modelList) {
          if (['classification', 'detection', 'segmentation', 'nlp'].includes(m.category)) {
            pretrained[m.category as keyof typeof pretrained][m.id] = {
              name: m.name,
              description: m.description,
              accuracy: m.accuracy,
              map50: m.map50,
              mIoU: m.mIoU,
              type: m.type,
            };
          } else {
            custom.push({
              name: m.name || m.id,
              filename: m.filename || m.id,
              path: m.path || '',
              format: m.format || 'pt',
              size: m.size || 0,
              uploaded: m.uploaded || '',
              metadata: m.metadata || {},
            });
          }
        }
        setModels({ pretrained, custom });
      }
    } catch (err) {
      console.error('Error loading models:', err);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setStatus('loading');
    setError(null);

    try {
      // First validate
      const validateResponse = await validateModel(selectedFile);
      if (!validateResponse.ok) {
        throw new Error('Model validation failed');
      }

      // Then upload
      const uploadResponse = await uploadModel(selectedFile);
      if (!uploadResponse.ok) {
        throw new Error('Model upload failed');
      }

      setStatus('success');
      setSelectedFile(null);
      await loadModels();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setStatus('error');
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto p-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-4">
              <Database className="w-8 h-8 text-cyan-400" />
              <h1 className="text-3xl font-bold text-white">模型管理</h1>
            </div>
            <p className="text-slate-400">
              Manage pretrained and custom models. Upload your own PyTorch (.pt, .pth) 
              or ONNX (.onnx) models for inference.
            </p>
          </div>

          {/* Upload Section */}
          <div className="card p-6 mb-8">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Upload size={20} />
              Upload Custom Model
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              <UploadZone
                onFileSelect={setSelectedFile}
                accept=".pt,.pth,.onnx"
                maxSize={500 * 1024 * 1024}
                disabled={status === 'loading'}
              />
              <div className="space-y-4">
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <h3 className="text-sm font-medium text-white mb-2">Supported Formats</h3>
                  <ul className="text-sm text-slate-400 space-y-1">
                    <li>.pt, .pth - PyTorch model checkpoints</li>
                    <li>.onnx - ONNX format for cross-platform inference</li>
                  </ul>
                </div>
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <h3 className="text-sm font-medium text-white mb-2">Maximum File Size</h3>
                  <p className="text-sm text-slate-400">500 MB</p>
                </div>
                {selectedFile && (
                  <button
                    onClick={handleUpload}
                    disabled={status === 'loading'}
                    className="btn-primary w-full"
                  >
                    {status === 'loading' ? 'Uploading...' : '上传模型'}
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Status */}
          {status !== 'idle' && (
            <StatusDisplay
              status={status}
              message={
                status === 'loading' ? 'Processing model...' :
                status === 'success' ? 'Model uploaded successfully!' :
                `Error: ${error}`
              }
            />
          )}

          {/* Pretrained Models */}
          {models && (
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Brain size={24} />
                Pretrained Models
              </h2>
              
              {/* Category Tabs */}
              <div className="flex gap-2 mb-6">
                {['classification', 'detection', 'segmentation', 'nlp'].map((category) => (
                  <button
                    key={category}
                    onClick={() => setActiveCategory(category)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium capitalize transition-all ${
                      activeCategory === category
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                        : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                    }`}
                  >
                    {category}
                  </button>
                ))}
              </div>

              {/* Models Grid */}
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(models.pretrained[activeCategory as keyof typeof models.pretrained] || {}).map(([key, model]) => (
                  <div key={key} className="card p-4">
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="text-white font-medium">{model.name}</h3>
                      <span className="badge badge-info">
                        {model.accuracy || model.map50 || model.mIoU || model.type || 'Ready'}
                      </span>
                    </div>
                    <p className="text-sm text-slate-400 mb-3">{model.description}</p>
                    <div className="flex items-center gap-2 text-xs text-slate-500">
                      <Info size={12} />
                      <span>ID: {key}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Custom Models */}
          {models && models.custom.length > 0 && (
            <div>
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Upload size={24} />
                Custom Models ({models.custom.length})
              </h2>
              
              <div className="space-y-4">
                {models.custom.map((model) => (
                  <div key={model.filename} className="card p-4 flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-3 mb-1">
                        <h3 className="text-white font-medium">{model.name}</h3>
                        <span className="badge badge-success uppercase">{model.format}</span>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-slate-400">
                        <span>{formatFileSize(model.size)}</span>
                        <span>Uploaded: {new Date(model.uploaded).toLocaleDateString()}</span>
                      </div>
                    </div>
                    <button className="p-2 hover:bg-red-500/20 rounded transition-colors text-red-400">
                      <Trash2 size={18} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
