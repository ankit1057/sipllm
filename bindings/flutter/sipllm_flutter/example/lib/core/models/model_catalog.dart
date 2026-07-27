import 'package:sipllm_flutter/sipllm_flutter.dart';

/// A downloadable GGUF model. Mirrors the SipLLM CLI registry plus larger/heavier
/// model classes designed to showcase SipLLM's core power: streaming large models
/// under tight RAM budgets.
class CatalogModel {
  const CatalogModel({
    required this.id,
    required this.name,
    required this.repo,
    required this.file,
    required this.quant,
    required this.approxBytes,
    required this.params,
    this.watchFriendly = false,
    this.recommended = false,
    this.revision = 'main',
  });

  /// Stable id (also the local filename stem).
  final String id;
  final String name;

  /// Hugging Face `<owner>/<name>` repo.
  final String repo;

  /// File within the repo.
  final String file;
  final String quant;

  /// Approximate on-disk size (for UI + expected-size verification hint).
  final int approxBytes;

  /// Human parameter count, e.g. "1.1B", "7B", "14B".
  final String params;

  /// Small enough to stream comfortably under a Wear OS RAM budget.
  final bool watchFriendly;
  final bool recommended;
  final String revision;

  String get url => HuggingFaceRepo(repo, revision: revision).resolveUrl(file);
  String get localFileName => '$id.gguf';

  double get approxMiB => approxBytes / (1024 * 1024);
  double get approxGiB => approxBytes / (1024 * 1024 * 1024);
}

/// The expanded catalog including compact, medium, and heavy models (135M to 14B).
const List<CatalogModel> kModelCatalog = [
  // --- Compact & Wear OS Friendly ---
  CatalogModel(
    id: 'smollm2-135m-q8',
    name: 'SmolLM2 135M Instruct',
    repo: 'bartowski/SmolLM2-135M-Instruct-GGUF',
    file: 'SmolLM2-135M-Instruct-Q8_0.gguf',
    quant: 'Q8_0',
    approxBytes: 145 * 1024 * 1024,
    params: '135M',
    watchFriendly: true,
    recommended: true,
  ),
  CatalogModel(
    id: 'smollm2-360m-q8',
    name: 'SmolLM2 360M Instruct',
    repo: 'bartowski/SmolLM2-360M-Instruct-GGUF',
    file: 'SmolLM2-360M-Instruct-Q8_0.gguf',
    quant: 'Q8_0',
    approxBytes: 386 * 1024 * 1024,
    params: '360M',
    watchFriendly: true,
  ),
  CatalogModel(
    id: 'llama3.2-1b-q4km',
    name: 'Llama 3.2 1B Instruct',
    repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF',
    file: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 808 * 1024 * 1024,
    params: '1B',
    recommended: true,
  ),
  CatalogModel(
    id: 'qwen2.5-1.5b-q4km',
    name: 'Qwen 2.5 1.5B Instruct',
    repo: 'Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    file: 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
    quant: 'Q4_K_M',
    approxBytes: 980 * 1024 * 1024,
    params: '1.5B',
  ),
  CatalogModel(
    id: 'smollm2-1.7b-q4km',
    name: 'SmolLM2 1.7B Instruct',
    repo: 'bartowski/SmolLM2-1.7B-Instruct-GGUF',
    file: 'SmolLM2-1.7B-Instruct-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 1060 * 1024 * 1024,
    params: '1.7B',
  ),

  // --- Medium Models (2B - 3.8B) ---
  CatalogModel(
    id: 'gemma2-2b-q4km',
    name: 'Gemma 2 2B Instruct',
    repo: 'bartowski/gemma-2-2b-it-GGUF',
    file: 'gemma-2-2b-it-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 1630 * 1024 * 1024,
    params: '2.6B',
    recommended: true,
  ),
  CatalogModel(
    id: 'qwen2.5-3b-q4km',
    name: 'Qwen 2.5 3B Instruct',
    repo: 'Qwen/Qwen2.5-3B-Instruct-GGUF',
    file: 'qwen2.5-3b-instruct-q4_k_m.gguf',
    quant: 'Q4_K_M',
    approxBytes: 1950 * 1024 * 1024,
    params: '3B',
  ),
  CatalogModel(
    id: 'llama3.2-3b-q4km',
    name: 'Llama 3.2 3B Instruct',
    repo: 'bartowski/Llama-3.2-3B-Instruct-GGUF',
    file: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 2020 * 1024 * 1024,
    params: '3.2B',
  ),
  CatalogModel(
    id: 'phi3.5-mini-q4km',
    name: 'Phi 3.5 Mini 3.8B Instruct',
    repo: 'bartowski/Phi-3.5-mini-instruct-GGUF',
    file: 'Phi-3.5-mini-instruct-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 2250 * 1024 * 1024,
    params: '3.8B',
  ),

  // --- Heavy Models (7B - 14B+) & Expanded Architectures ---
  CatalogModel(
    id: 'kimi-k1.5-7b-q4km',
    name: 'Kimi K1.5 7B Instruct (Moonshot)',
    repo: 'bartowski/Kimi-k1.5-7B-Instruct-GGUF',
    file: 'Kimi-k1.5-7B-Instruct-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 4350 * 1024 * 1024,
    params: '7.2B',
    recommended: true,
  ),
  CatalogModel(
    id: 'gemma4-9b-q4km',
    name: 'Gemma 4 9B Instruct',
    repo: 'bartowski/gemma-4-9b-it-GGUF',
    file: 'gemma-4-9b-it-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 5500 * 1024 * 1024,
    params: '9B',
    recommended: true,
  ),
  CatalogModel(
    id: 'deepseek-r1-distill-qwen-7b-q4km',
    name: 'DeepSeek R1 Distill Qwen 7B',
    repo: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF',
    file: 'DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 4420 * 1024 * 1024,
    params: '7.6B',
    recommended: true,
  ),
  CatalogModel(
    id: 'yi-1.5-9b-q4km',
    name: 'Yi 1.5 9B Chat',
    repo: '01-ai/Yi-1.5-9B-Chat-GGUF',
    file: 'Yi-1.5-9B-Chat-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 5300 * 1024 * 1024,
    params: '8.8B',
  ),
  CatalogModel(
    id: 'baichuan2-7b-q4km',
    name: 'Baichuan 2 7B Chat',
    repo: 'baichuan-inc/Baichuan2-7B-Chat-GGUF',
    file: 'Baichuan2-7B-Chat-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 4200 * 1024 * 1024,
    params: '7B',
  ),
  CatalogModel(
    id: 'internlm2.5-7b-q4km',
    name: 'InternLM 2.5 7B Chat',
    repo: 'internlm/internlm2_5-7b-chat-gguf',
    file: 'internlm2_5-7b-chat-q4_k_m.gguf',
    quant: 'Q4_K_M',
    approxBytes: 4400 * 1024 * 1024,
    params: '7.7B',
  ),
  CatalogModel(
    id: 'glm4-9b-q4km',
    name: 'GLM-4 9B Chat (Zhipu)',
    repo: 'THUDM/glm-4-9b-chat-GGUF',
    file: 'glm-4-9b-chat-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 5450 * 1024 * 1024,
    params: '9B',
  ),
  CatalogModel(
    id: 'phi4-14b-q4km',
    name: 'Phi-4 14B Instruct',
    repo: 'microsoft/phi-4-GGUF',
    file: 'phi-4-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 8900 * 1024 * 1024,
    params: '14.7B',
    recommended: true,
  ),
  CatalogModel(
    id: 'mistral7b-q4km',
    name: 'Mistral 7B Instruct v0.3',
    repo: 'bartowski/Mistral-7B-Instruct-v0.3-GGUF',
    file: 'Mistral-7B-Instruct-v0.3-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 4140 * 1024 * 1024,
    params: '7.2B',
  ),
  CatalogModel(
    id: 'qwen2.5-7b-q4km',
    name: 'Qwen 2.5 7B Instruct',
    repo: 'Qwen/Qwen2.5-7B-Instruct-GGUF',
    file: 'qwen2.5-7b-instruct-q4_k_m.gguf',
    quant: 'Q4_K_M',
    approxBytes: 4400 * 1024 * 1024,
    params: '7.6B',
  ),
  CatalogModel(
    id: 'llama3.1-8b-q4km',
    name: 'Llama 3.1 8B Instruct',
    repo: 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF',
    file: 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 4920 * 1024 * 1024,
    params: '8B',
  ),
  CatalogModel(
    id: 'gemma2-9b-q4km',
    name: 'Gemma 2 9B Instruct',
    repo: 'bartowski/gemma-2-9b-it-GGUF',
    file: 'gemma-2-9b-it-Q4_K_M.gguf',
    quant: 'Q4_K_M',
    approxBytes: 5400 * 1024 * 1024,
    params: '9.2B',
  ),
  CatalogModel(
    id: 'qwen2.5-14b-q4km',
    name: 'Qwen 2.5 14B Instruct (Heavy)',
    repo: 'Qwen/Qwen2.5-14B-Instruct-GGUF',
    file: 'qwen2.5-14b-instruct-q4_k_m.gguf',
    quant: 'Q4_K_M',
    approxBytes: 8980 * 1024 * 1024,
    params: '14.7B',
  ),
];
