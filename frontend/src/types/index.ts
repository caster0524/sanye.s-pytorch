// API Types for PyTorch Deep Learning Studio

export interface UploadResponse {
  success: boolean;
  filename: string;
  saved_as: string;
  path: string;
  size: number;
  type: string;
}

export interface ModelInfo {
  name: string;
  description: string;
  input_size?: [number, number];
  accuracy?: string;
  map50?: string;
  mIoU?: string;
  type?: string;
}

export interface ClassificationPrediction {
  class: string;
  class_id: number;
  probability: number;
}

export interface ClassificationResult {
  success: boolean;
  model: string;
  model_info: ModelInfo;
  predictions: ClassificationPrediction[];
  image_size: [number, number];
}

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
}

export interface DetectionResult {
  success: boolean;
  model: string;
  detections: Detection[];
  count: number;
  annotated_image: string;
}

export interface SegmentationResult {
  success: boolean;
  model: string;
  original_size: [number, number];
  mask_size: [number, number];
  mask: string;
  overlay: string;
  class_distribution: Record<string, number>;
  num_classes: number;
}

export interface SentimentResult {
  success: boolean;
  model: string;
  text: string;
  sentiment: string;
  sentiment_label?: string;
  confidence: number;
  raw_label: string;
}

export interface GenerationResult {
  success: boolean;
  prompt: string;
  negative_prompt?: string;
  parameters: {
    width: number;
    height: number;
    steps: number;
    guidance_scale: number;
    seed: string | number;
  };
  image_path?: string;
  image_base64?: string;
  demo?: boolean;
  message?: string;
}

export interface TrainingConfig {
  model_name: string;
  dataset_path: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  num_classes: number;
  optimizer: string;
  scheduler: string;
}

export interface TrainingJob {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_epoch: number;
  total_epochs: number;
  loss: number;
  accuracy: number;
  start_time: string | null;
  end_time: string | null;
  error: string | null;
  model_path?: string;
}

export interface CustomModel {
  name: string;
  filename: string;
  path: string;
  format: string;
  size: number;
  uploaded: string;
  metadata: Record<string, unknown>;
}

export interface ModelsResponse {
  pretrained: {
    classification: Record<string, ModelInfo>;
    detection: Record<string, ModelInfo>;
    segmentation: Record<string, ModelInfo>;
    nlp: Record<string, ModelInfo>;
  };
  custom: CustomModel[];
}

export type TaskType = 'classification' | 'detection' | 'segmentation' | 'nlp' | 'generation' | 'training';
