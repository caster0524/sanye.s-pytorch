'use client';

import { useState } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface ModelSelectorProps {
  models: Record<string, { name: string; description: string; input_size?: [number, number]; accuracy?: string }>;
  selectedModel: string;
  onModelChange: (model: string) => void;
  label?: string;
}

export default function ModelSelector({
  models,
  selectedModel,
  onModelChange,
  label = '选择模型',
}: ModelSelectorProps) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-slate-300">{label}</label>
      <select
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        className="input-field"
      >
        {Object.entries(models).map(([key, model]) => (
          <option key={key} value={key}>
            {model.name} {model.accuracy ? `(${model.accuracy})` : ''}
          </option>
        ))}
      </select>
      <p className="text-xs text-slate-500">
        {models[selectedModel]?.description || '请选择模型'}
      </p>
    </div>
  );
}

interface SliderInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
}

export function SliderInput({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  unit = '',
}: SliderInputProps) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between">
        <label className="text-sm font-medium text-slate-300">{label}</label>
        <span className="text-sm text-slate-400">{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
      />
    </div>
  );
}

interface NumberInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

export function NumberInput({
  label,
  value,
  onChange,
  min = 0,
  max = 1000,
  step = 1,
}: NumberInputProps) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-slate-300">{label}</label>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="input-field"
      />
    </div>
  );
}
