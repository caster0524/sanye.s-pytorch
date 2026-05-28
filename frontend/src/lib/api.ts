// API Client for PyTorch Deep Learning Studio

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = 60000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

// File Upload
export async function uploadFile(file: File): Promise<{ success: boolean; path?: string; error?: string }> {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetchWithTimeout(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Upload error:', error);
    return { success: false, error: String(error) };
  }
}

// Image Classification
export async function classifyImage(
  file: File,
  modelName: string = 'resnet50',
  topK: number = 5
): Promise<Response> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', modelName);
  formData.append('top_k', String(topK));
  
  return fetchWithTimeout(`${API_BASE}/api/classify`, {
    method: 'POST',
    body: formData,
  });
}

// Object Detection
export async function detectObjects(
  file: File,
  modelName: string = 'faster_rcnn',
  confidenceThreshold: number = 0.5
): Promise<Response> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', modelName);
  formData.append('confidence_threshold', String(confidenceThreshold));
  
  return fetchWithTimeout(`${API_BASE}/api/detect`, {
    method: 'POST',
    body: formData,
  });
}

// Image Segmentation
export async function segmentImage(
  file: File,
  modelName: string = 'deeplabv3_resnet50',
  overlayAlpha: number = 0.5
): Promise<Response> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', modelName);
  formData.append('overlay_alpha', String(overlayAlpha));
  
  return fetchWithTimeout(`${API_BASE}/api/segment`, {
    method: 'POST',
    body: formData,
  });
}

// NLP - Sentiment Analysis
export async function analyzeSentiment(text: string, model: string = 'bert_sentiment'): Promise<Response> {
  return fetchWithTimeout(`${API_BASE}/api/nlp/sentiment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model }),
  });
}

// NLP - Text Generation
export async function generateText(
  prompt: string,
  maxLength: number = 100,
  temperature: number = 0.9,
  model: string = 'gpt2_text'
): Promise<Response> {
  return fetchWithTimeout(`${API_BASE}/api/nlp/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, max_length: maxLength, temperature, top_p: 0.9, model }),
  });
}

// NLP - Summarization
export async function summarizeText(text: string): Promise<Response> {
  return fetchWithTimeout(`${API_BASE}/api/nlp/summarize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
}

// Image Generation
export async function generateImage(
  prompt: string,
  negativePrompt: string = '',
  width: number = 512,
  height: number = 512,
  steps: number = 30,
  guidanceScale: number = 7.5,
  seed: number = -1
): Promise<Response> {
  return fetchWithTimeout(`${API_BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      negative_prompt: negativePrompt,
      width,
      height,
      steps,
      guidance_scale: guidanceScale,
      seed,
    }),
  });
}

// Model Management
export async function getModels(): Promise<Response> {
  return fetchWithTimeout(`${API_BASE}/api/models`);
}

export async function uploadModel(file: File, metadata?: Record<string, unknown>): Promise<Response> {
  const formData = new FormData();
  formData.append('file', file);
  if (metadata) {
    formData.append('metadata', JSON.stringify(metadata));
  }
  
  return fetchWithTimeout(`${API_BASE}/api/models/upload`, {
    method: 'POST',
    body: formData,
  });
}

export async function validateModel(file: File): Promise<Response> {
  const formData = new FormData();
  formData.append('file', file);
  
  return fetchWithTimeout(`${API_BASE}/api/models/validate`, {
    method: 'POST',
    body: formData,
  });
}

// Training
export async function startTraining(config: {
  model_name: string;
  dataset_path?: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  num_classes: number;
  optimizer: string;
  scheduler: string;
}): Promise<Response> {
  return fetchWithTimeout(`${API_BASE}/api/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
}

export async function getTrainingStatus(jobId: string): Promise<Response> {
  return fetchWithTimeout(`${API_BASE}/api/train/status/${jobId}`);
}

export async function getTrainingJobs(): Promise<Response> {
  return fetchWithTimeout(`${API_BASE}/api/train/jobs`);
}

export async function getTrainingHistory(): Promise<Response> {
  return fetchWithTimeout(`${API_BASE}/api/train/history`);
}

// Health Check
export async function healthCheck(): Promise<{ status: string; device: string }> {
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    return await response.json();
  } catch {
    return { status: 'unavailable', device: 'unknown' };
  }
}
