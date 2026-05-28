'use client';

import { useState, useEffect } from 'react';
import Sidebar from '@/components/Sidebar';
import StatusDisplay from '@/components/StatusDisplay';
import { startTraining, getTrainingStatus, getTrainingJobs } from '@/lib/api';
import { Brain, Play, Clock, CheckCircle, XCircle } from 'lucide-react';

const trainingModels = {
  resnet50: {
    name: 'ResNet50',
    description: '用于图像分类的残差网络',
  },
  vgg16: {
    name: 'VGG16',
    description: '视觉几何组网络',
  },
};

export default function TrainingPage() {
  const [modelName, setModelName] = useState('resnet50');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [numClasses, setNumClasses] = useState(10);
  const [optimizer, setOptimizer] = useState('adam');
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Record<string, unknown>[]>([]);
  const [currentJob, setCurrentJob] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    loadJobs();
  }, []);

  useEffect(() => {
    if (jobId) {
      const interval = setInterval(async () => {
        const res = await getTrainingStatus(jobId);
        const data = await res.json();
        setCurrentJob(data);
        if (data.status === 'completed' || data.status === 'error') {
          clearInterval(interval);
          setStatus(data.status === 'completed' ? 'success' : 'error');
        }
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [jobId]);

  const loadJobs = async () => {
    try {
      const res = await getTrainingJobs();
      if (res.ok) {
        const data = await res.json();
        setJobs(data.jobs || []);
      }
    } catch (err) {
      console.error('Failed to load jobs:', err);
    }
  };

  const handleStartTraining = async () => {
    setStatus('loading');
    try {
      const res = await startTraining({
        model_name: modelName,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        num_classes: numClasses,
        optimizer,
        scheduler: ' cosine',
      });
      
      if (res.ok) {
        const data = await res.json();
        setJobId(data.job_id);
      } else {
        setStatus('error');
      }
    } catch {
      setStatus('error');
    }
  };

  const getStatusIcon = (jobStatus: string) => {
    switch (jobStatus) {
      case 'completed':
        return <CheckCircle className="text-green-400" size={16} />;
      case 'error':
        return <XCircle className="text-red-400" size={16} />;
      default:
        return <Clock className="text-blue-400 animate-pulse" size={16} />;
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
              <Brain className="w-8 h-8 text-indigo-400" />
              <h1 className="text-3xl font-bold text-white">模型训练</h1>
            </div>
            <p className="text-slate-400">
              在您自己的数据集上训练自定义模型。配置训练参数并监控进度。
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Configuration */}
            <div className="space-y-6">
              <div className="card p-6">
                <h2 className="text-lg font-semibold text-white mb-4">配置</h2>
                
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-slate-300 block mb-2">模型类型</label>
                    <select
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      className="input-field"
                    >
                      {Object.entries(trainingModels).map(([key, model]) => (
                        <option key={key} value={key}>{model.name}</option>
                      ))}
                    </select>
                    <p className="text-xs text-slate-500 mt-1">
                      {trainingModels[modelName as keyof typeof trainingModels]?.description}
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium text-slate-300 block mb-2">训练轮数</label>
                      <input
                        type="number"
                        value={epochs}
                        onChange={(e) => setEpochs(Number(e.target.value))}
                        min={1}
                        max={100}
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium text-slate-300 block mb-2">批次大小</label>
                      <input
                        type="number"
                        value={batchSize}
                        onChange={(e) => setBatchSize(Number(e.target.value))}
                        min={1}
                        max={256}
                        className="input-field"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-slate-300 block mb-2">
                      学习率: {learningRate}
                    </label>
                    <input
                      type="range"
                      value={learningRate}
                      onChange={(e) => setLearningRate(Number(e.target.value))}
                      min={0.0001}
                      max={0.1}
                      step={0.0001}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium text-slate-300 block mb-2">类别数量</label>
                    <input
                      type="number"
                      value={numClasses}
                      onChange={(e) => setNumClasses(Number(e.target.value))}
                      min={2}
                      max={1000}
                      className="input-field"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium text-slate-300 block mb-2">优化器</label>
                    <select
                      value={optimizer}
                      onChange={(e) => setOptimizer(e.target.value)}
                      className="input-field"
                    >
                      <option value="adam">Adam</option>
                      <option value="sgd">SGD</option>
                      <option value="adamw">AdamW</option>
                    </select>
                  </div>
                </div>
              </div>

              <button
                onClick={handleStartTraining}
                disabled={status === 'loading'}
                className="btn-primary w-full py-3 flex items-center justify-center gap-2"
              >
                {status === 'loading' ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    训练中...
                  </>
                ) : (
                  <>
                    <Play size={20} />
                    开始训练
                  </>
                )}
              </button>

              {status !== 'idle' && (
                <StatusDisplay
                  status={status}
                  message={
                    status === 'loading' ? '模型训练中...' :
                    status === 'success' ? '训练完成！' :
                    '训练出错'
                  }
                />
              )}

              {currentJob && (
                <div className="card p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">当前训练进度</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-400">状态</span>
                      <span className="text-white capitalize">{currentJob.status as string}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">当前轮数</span>
                      <span className="text-white">{currentJob.current_epoch as number}/{epochs}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">损失值</span>
                      <span className="text-white">{(currentJob.loss as number)?.toFixed(4) || '-'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">准确率</span>
                      <span className="text-white">{((currentJob.accuracy as number) * 100)?.toFixed(2) || '-'}%</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Training Jobs */}
            <div className="space-y-6">
              <div className="card p-6">
                <h2 className="text-lg font-semibold text-white mb-4">训练任务</h2>
                
                {jobs.length === 0 ? (
                  <div className="text-center py-8 text-slate-500">
                    <Clock className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>暂无训练任务</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {jobs.map((job) => (
                      <div
                        key={job.job_id as string}
                        className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg"
                      >
                        <div className="flex items-center gap-3">
                          {getStatusIcon(job.status as string)}
                          <div>
                            <p className="text-white font-medium">{job.model_type as string}</p>
                            <p className="text-xs text-slate-500">
                              {new Date(job.created_at as string).toLocaleString('zh-CN')}
                            </p>
                          </div>
                        </div>
                        <span className="text-sm text-slate-400">
                          {job.status as string}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
