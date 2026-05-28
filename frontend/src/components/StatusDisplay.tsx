'use client';

import { Loader2, CheckCircle, XCircle, AlertCircle, Info } from 'lucide-react';

interface StatusDisplayProps {
  status: 'idle' | 'loading' | 'success' | 'error';
  message?: string;
  progress?: number;
  autoHide?: boolean;
  onHide?: () => void;
}

export default function StatusDisplay({
  status,
  message,
  progress,
  autoHide = false,
  onHide,
}: StatusDisplayProps) {
  if (status === 'idle') return null;

  const statusConfig = {
    loading: {
      icon: <Loader2 className="animate-spin" size={20} />,
      bgClass: 'bg-blue-500/10',
      borderClass: 'border-blue-500/30',
      textClass: 'text-blue-400',
    },
    success: {
      icon: <CheckCircle size={20} />,
      bgClass: 'bg-green-500/10',
      borderClass: 'border-green-500/30',
      textClass: 'text-green-400',
    },
    error: {
      icon: <XCircle size={20} />,
      bgClass: 'bg-red-500/10',
      borderClass: 'border-red-500/30',
      textClass: 'text-red-400',
    },
  };

  const config = statusConfig[status];

  return (
    <div
      className={`
        rounded-lg border p-4 animate-fadeIn
        ${config.bgClass} ${config.borderClass}
      `}
    >
      <div className="flex items-center gap-3">
        <span className={config.textClass}>{config.icon}</span>
        <div className="flex-1">
          {message && (
            <p className={`text-sm ${config.textClass}`}>{message}</p>
          )}
          {status === 'loading' && progress !== undefined && (
            <div className="mt-2">
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-xs text-slate-400 mt-1">
                {progress.toFixed(0)}% complete
              </p>
            </div>
          )}
        </div>
        {onHide && (
          <button
            onClick={onHide}
            className="text-slate-400 hover:text-white transition-colors"
          >
            &times;
          </button>
        )}
      </div>
    </div>
  );
}

// Toast notification component
interface ToastProps {
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  onClose: () => void;
}

export function Toast({ type, message, onClose }: ToastProps) {
  const config = {
    success: {
      icon: <CheckCircle size={18} />,
      bgClass: 'bg-green-500/20 border-green-500/30',
      textClass: 'text-green-400',
    },
    error: {
      icon: <XCircle size={18} />,
      bgClass: 'bg-red-500/20 border-red-500/30',
      textClass: 'text-red-400',
    },
    warning: {
      icon: <AlertCircle size={18} />,
      bgClass: 'bg-yellow-500/20 border-yellow-500/30',
      textClass: 'text-yellow-400',
    },
    info: {
      icon: <Info size={18} />,
      bgClass: 'bg-blue-500/20 border-blue-500/30',
      textClass: 'text-blue-400',
    },
  };

  const c = config[type];

  return (
    <div
      className={`
        fixed bottom-4 right-4 z-50 rounded-lg border p-4 
        flex items-center gap-3 animate-fadeIn shadow-lg
        ${c.bgClass}
      `}
      style={{ minWidth: '300px' }}
    >
      <span className={c.textClass}>{c.icon}</span>
      <p className="text-sm text-slate-200 flex-1">{message}</p>
      <button
        onClick={onClose}
        className="text-slate-400 hover:text-white transition-colors"
      >
        &times;
      </button>
    </div>
  );
}
