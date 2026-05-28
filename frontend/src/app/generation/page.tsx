'use client';

import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import StatusDisplay from '@/components/StatusDisplay';
import { generateImage } from '@/lib/api';
import { Sparkles, RefreshCw } from 'lucide-react';

export default function GenerationPage() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [steps, setSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState(-1);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setStatus('loading');
    setError(null);
    setResult(null);

    try {
      const response = await generateImage(
        prompt,
        negativePrompt,
        width,
        height,
        steps,
        guidanceScale,
        seed
      );
      
      if (!response.ok) throw new Error('Generation failed');
      const data = await response.json();
      setResult(data);
      setStatus('success');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setStatus('error');
    }
  };

  const handleRandomSeed = () => {
    setSeed(Math.floor(Math.random() * 2147483647));
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto p-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-4">
              <Sparkles className="w-8 h-8 text-pink-400" />
              <h1 className="text-3xl font-bold text-white">AI 图像生成</h1>
            </div>
            <p className="text-slate-400">
              根据文本提示生成图像 using Stable Diffusion. 
              Create stunning artwork, illustrations, and photorealistic images.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Input Section */}
            <div className="space-y-6">
              {/* Prompt */}
              <div className="card p-6">
                <h2 className="text-lg font-semibold text-white mb-4">Prompt</h2>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="A beautiful landscape with mountains and a lake at sunset, digital art, highly detailed..."
                  className="input-field w-full h-32 resize-none"
                  disabled={status === 'loading'}
                />
              </div>

              {/* Negative Prompt */}
              <div className="card p-6">
                <h2 className="text-lg font-semibold text-white mb-4">Negative Prompt (Optional)</h2>
                <textarea
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  placeholder="blur, low quality, distorted, ugly..."
                  className="input-field w-full h-24 resize-none"
                  disabled={status === 'loading'}
                />
              </div>

              {/* Settings */}
              <div className="card p-6">
                <h2 className="text-lg font-semibold text-white mb-4">生成设置</h2>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <label className="text-sm font-medium text-slate-300">Width</label>
                      <span className="text-sm text-slate-400">{width}px</span>
                    </div>
                    <input
                      type="range"
                      min="256"
                      max="1024"
                      step="64"
                      value={width}
                      onChange={(e) => setWidth(Number(e.target.value))}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <label className="text-sm font-medium text-slate-300">Height</label>
                      <span className="text-sm text-slate-400">{height}px</span>
                    </div>
                    <input
                      type="range"
                      min="256"
                      max="1024"
                      step="64"
                      value={height}
                      onChange={(e) => setHeight(Number(e.target.value))}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <label className="text-sm font-medium text-slate-300">Steps</label>
                      <span className="text-sm text-slate-400">{steps}</span>
                    </div>
                    <input
                      type="range"
                      min="10"
                      max="100"
                      value={steps}
                      onChange={(e) => setSteps(Number(e.target.value))}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <label className="text-sm font-medium text-slate-300">Guidance</label>
                      <span className="text-sm text-slate-400">{guidanceScale}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="20"
                      step="0.5"
                      value={guidanceScale}
                      onChange={(e) => setGuidanceScale(Number(e.target.value))}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between mb-2">
                    <label className="text-sm font-medium text-slate-300">Seed</label>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-slate-400">{seed}</span>
                      <button
                        onClick={handleRandomSeed}
                        className="p-1 hover:bg-slate-700 rounded transition-colors"
                        title="Random seed"
                      >
                        <RefreshCw size={14} className="text-slate-400" />
                      </button>
                    </div>
                  </div>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                    placeholder="-1 for random"
                    className="input-field"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Set to -1 for random seed. Same seed + prompt = same image.
                  </p>
                </div>
              </div>

              {/* Actions */}
              <button
                onClick={handleGenerate}
                disabled={!prompt.trim() || status === 'loading'}
                className="btn-primary w-full py-3 flex items-center justify-center gap-2"
              >
                {status === 'loading' ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    生成中...
                  </>
                ) : (
                  <>
                    <Sparkles size={20} />
                    生成图像
                  </>
                )}
              </button>

              {status !== 'idle' && (
                <StatusDisplay
                  status={status}
                  message={
                    status === 'loading' ? '生成中...age (this may take a while)...' :
                    status === 'success' ? 'Image generated!' :
                    `Error: ${error}`
                  }
                />
              )}
            </div>

            {/* Results Section */}
            <div className="space-y-6">
              {result ? (
                <div className="card p-6">
                  <h2 className="text-lg font-semibold text-white mb-4">Generated Image</h2>
                  {(result as { image_base64?: string }).image_base64 ? (
                    <img
                      src={(result as { image_base64: string }).image_base64}
                      alt="Generated"
                      className="w-full rounded-lg"
                    />
                  ) : (
                    <div className="bg-slate-800/50 p-4 rounded-lg">
                      <p className="text-slate-400">Demo Mode</p>
                      <p className="text-slate-300 mt-2">
                        {(result as { message?: string }).message || 'Generation requires GPU resources.'}
                      </p>
                    </div>
                  )}
                  
                  {/* Parameters */}
                  <div className="mt-4 p-4 bg-slate-800/50 rounded-lg">
                    <h3 className="text-sm font-medium text-slate-300 mb-2">Parameters Used</h3>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      {Object.entries((result as { parameters: Record<string, unknown> }).parameters || {}).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                          <span className="text-slate-500">{key}:</span>
                          <span className="text-slate-300">{String(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="card p-12 flex flex-col items-center justify-center text-center">
                  <Sparkles className="w-16 h-16 text-slate-600 mb-4" />
                  <p className="text-slate-400">
                    Enter a prompt and click Generate to create an image
                  </p>
                  <p className="text-slate-500 text-sm mt-2">
                    Note: Image generation requires GPU resources
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
