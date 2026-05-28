'use client';

import { useState, useCallback } from 'react';
import { Upload, X, Image as ImageIcon, Loader2 } from 'lucide-react';

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  maxSize?: number;
  disabled?: boolean;
}

export default function UploadZone({
  onFileSelect,
  accept = 'image/*',
  maxSize = 10 * 1024 * 1024,
  disabled = false,
}: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (disabled) return;
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      validateAndSetFile(droppedFile);
    }
  }, [disabled]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      validateAndSetFile(selectedFile);
    }
  }, []);

  const validateAndSetFile = (selectedFile: File) => {
    setError(null);
    
    // Check file size
    if (selectedFile.size > maxSize) {
      setError(`文件大小超过 ${Math.round(maxSize / 1024 / 1024)}MB 限制`);
      return;
    }
    
    // Check file type
    if (accept !== '*/*' && !selectedFile.type.startsWith(accept.replace('/*', ''))) {
      setError('无效的文件类型');
      return;
    }
    
    setFile(selectedFile);
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(selectedFile);
    
    onFileSelect(selectedFile);
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setError(null);
  };

  if (preview) {
    return (
      <div className="relative">
        <div className="card p-2">
          <img
            src={preview}
            alt="Preview"
            className="w-full h-64 object-contain rounded-lg"
          />
          <div className="mt-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ImageIcon size={16} className="text-slate-400" />
              <span className="text-sm text-slate-300 truncate max-w-[200px]">
                {file?.name}
              </span>
              <span className="text-xs text-slate-500">
                ({((file?.size || 0) / 1024 / 1024).toFixed(2)} MB)
              </span>
            </div>
            <button
              onClick={handleClear}
              className="p-1 hover:bg-slate-600 rounded transition-colors"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`
        upload-zone relative
        ${isDragging ? 'dragging' : ''}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <input
        type="file"
        accept={accept}
        onChange={handleFileInput}
        disabled={disabled}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
      />
      
      <div className="flex flex-col items-center gap-3">
        {disabled ? (
          <Loader2 size={48} className="text-slate-500 animate-spin" />
        ) : (
          <Upload size={48} className="text-slate-500" />
        )}
        
        <div className="text-center">
          <p className="text-slate-300 font-medium">
            {disabled ? '处理中...' : '拖拽或点击上传文件'}
          </p>
          <p className="text-sm text-slate-500 mt-1">
            {disabled ? '请稍候' : `支持格式: 图片 (最大 ${Math.round(maxSize / 1024 / 1024)}MB)`}
          </p>
        </div>
      </div>
      
      {error && (
        <div className="absolute bottom-2 left-0 right-0 text-center">
          <span className="text-sm text-red-400">{error}</span>
        </div>
      )}
    </div>
  );
}
