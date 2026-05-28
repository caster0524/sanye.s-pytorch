'use client';

import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import StatusDisplay from '@/components/StatusDisplay';
import { analyzeSentiment, generateText, summarizeText } from '@/lib/api';
import { FileText, Send, Sparkles, BookOpen } from 'lucide-react';
import type { SentimentResult } from '@/types';

export default function NLPPage() {
  const [activeTab, setActiveTab] = useState<'sentiment' | 'generate' | 'summarize'>('sentiment');
  const [text, setText] = useState('');
  const [maxLength, setMaxLength] = useState(100);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSentiment = async () => {
    if (!text.trim()) return;
    setStatus('loading');
    setError(null);
    setResult(null);

    try {
      const response = await analyzeSentiment(text);
      if (!response.ok) throw new Error('Analysis failed');
      const data = await response.json();
      // Backend nests data in data.result, flatten for UI
      setResult({ ...data.result, text: data.input_text || text });
      setStatus('success');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setStatus('error');
    }
  };

  const handleGenerate = async () => {
    if (!text.trim()) return;
    setStatus('loading');
    setError(null);
    setResult(null);

    try {
      const response = await generateText(text, maxLength);
      if (!response.ok) throw new Error('生成中...iled');
      const data = await response.json();
      // Backend nests data in data.result
      setResult(data.result);
      setStatus('success');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setStatus('error');
    }
  };

  const handleSummarize = async () => {
    if (!text.trim()) return;
    setStatus('loading');
    setError(null);
    setResult(null);

    try {
      const response = await summarizeText(text);
      if (!response.ok) throw new Error('Summarization failed');
      const data = await response.json();
      // Backend nests data in data.result; add original_text for UI
      setResult({ ...data.result, original_text: text });
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
              <FileText className="w-8 h-8 text-orange-400" />
              <h1 className="text-3xl font-bold text-white">自然语言处理</h1>
            </div>
            <p className="text-slate-400">
              Process and analyze text using state-of-the-art NLP models. 
              Sentiment analysis, text generation, and summarization powered by BERT and GPT-2.
            </p>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => { setActiveTab('sentiment'); setResult(null); setStatus('idle'); }}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                activeTab === 'sentiment'
                  ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              情感分析
            </button>
            <button
              onClick={() => { setActiveTab('generate'); setResult(null); setStatus('idle'); }}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                activeTab === 'generate'
                  ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              文本生成
            </button>
            <button
              onClick={() => { setActiveTab('summarize'); setResult(null); setStatus('idle'); }}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                activeTab === 'summarize'
                  ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              Summarization
            </button>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Input Section */}
            <div className="space-y-6">
              <div className="card p-6">
                <h2 className="text-lg font-semibold text-white mb-4">
                  {activeTab === 'sentiment' && 'Enter Text for 情感分析'}
                  {activeTab === 'generate' && 'Enter Prompt for 文本生成'}
                  {activeTab === 'summarize' && 'Enter Text to Summarize'}
                </h2>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder={
                    activeTab === 'sentiment' 
                      ? 'Enter text to analyze sentiment...'
                      : activeTab === 'generate'
                      ? 'Enter a prompt to generate text...'
                      : 'Enter text to summarize...'
                  }
                  className="input-field w-full h-40 resize-none"
                  disabled={status === 'loading'}
                />
                
                {activeTab === 'generate' && (
                  <div className="mt-4">
                    <div className="flex justify-between">
                      <label className="text-sm font-medium text-slate-300">
                        Max Length
                      </label>
                      <span className="text-sm text-slate-400">{maxLength} tokens</span>
                    </div>
                    <input
                      type="range"
                      min="20"
                      max="300"
                      value={maxLength}
                      onChange={(e) => setMaxLength(Number(e.target.value))}
                      className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500 mt-2"
                    />
                  </div>
                )}
              </div>

              <button
                onClick={activeTab === 'sentiment' ? handleSentiment : activeTab === 'generate' ? handleGenerate : handleSummarize}
                disabled={!text.trim() || status === 'loading'}
                className="btn-primary w-full py-3 flex items-center justify-center gap-2"
              >
                {status === 'loading' ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Send size={20} />
                    {activeTab === 'sentiment' && 'Analyze Sentiment'}
                    {activeTab === 'generate' && 'Generate Text'}
                    {activeTab === 'summarize' && 'Summarize'}
                  </>
                )}
              </button>

              {status !== 'idle' && (
                <StatusDisplay
                  status={status}
                  message={
                    status === 'loading' ? 'Processing...' :
                    status === 'success' ? 'Complete!' :
                    `Error: ${error}`
                  }
                />
              )}
            </div>

            {/* Results Section */}
            <div className="space-y-6">
              {result ? (
                <>
                  {activeTab === 'sentiment' && (
                    <div className="card p-6">
                      <h2 className="text-lg font-semibold text-white mb-4">Sentiment Result</h2>
                      <div className="flex items-center gap-4 mb-6">
                        <div className={`px-4 py-2 rounded-lg text-lg font-bold ${
                          (result as unknown as SentimentResult).sentiment.includes('positive') 
                            ? 'bg-green-500/20 text-green-400'
                            : (result as unknown as SentimentResult).sentiment.includes('negative')
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-yellow-500/20 text-yellow-400'
                        }`}>
                          {(result as unknown as SentimentResult).sentiment_label || (result as unknown as SentimentResult).sentiment}
                        </div>
                        <div className="text-slate-400">
                          Confidence: <span className="text-white font-mono">
                            {((result as unknown as SentimentResult).confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      <div className="bg-slate-800/50 p-4 rounded-lg">
                        <p className="text-slate-400 text-sm mb-2">Original Text:</p>
                        <p className="text-slate-200">{(result as unknown as SentimentResult).text}</p>
                      </div>
                    </div>
                  )}

                  {activeTab === 'generate' && (
                    <div className="card p-6">
                      <h2 className="text-lg font-semibold text-white mb-4">
                        <Sparkles className="inline w-5 h-5 mr-2 text-orange-400" />
                        Generated Text
                      </h2>
                      <div className="bg-slate-800/50 p-4 rounded-lg mb-4">
                        <p className="text-slate-400 text-sm mb-2">Prompt:</p>
                        <p className="text-slate-200">{(result as { prompt: string }).prompt}</p>
                      </div>
                      <div className="bg-gradient-to-r from-orange-500/10 to-purple-500/10 p-4 rounded-lg border border-orange-500/20">
                        <p className="text-slate-400 text-sm mb-2">Generated:</p>
                        <p className="text-slate-200 whitespace-pre-wrap">{(result as { generated_text: string }).generated_text}</p>
                      </div>
                    </div>
                  )}

                  {activeTab === 'summarize' && (
                    <div className="card p-6">
                      <h2 className="text-lg font-semibold text-white mb-4">
                        <BookOpen className="inline w-5 h-5 mr-2 text-orange-400" />
                        Summary
                      </h2>
                      <div className="bg-slate-800/50 p-4 rounded-lg mb-4">
                        <p className="text-slate-400 text-sm mb-2">Original:</p>
                        <p className="text-slate-300 text-sm">{(result as { original_text: string }).original_text}</p>
                      </div>
                      <div className="bg-gradient-to-r from-orange-500/10 to-yellow-500/10 p-4 rounded-lg border border-orange-500/20">
                        <p className="text-slate-400 text-sm mb-2">Summary:</p>
                        <p className="text-slate-200 font-medium">{(result as { summary: string }).summary}</p>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="card p-12 flex flex-col items-center justify-center text-center">
                  <FileText className="w-16 h-16 text-slate-600 mb-4" />
                  <p className="text-slate-400">
                    Enter text and click the button to see results
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
