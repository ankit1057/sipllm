/// SipLLM for Flutter — FFI bindings to the streaming, memory-bounded LLM engine
/// that runs models larger than RAM, plus a resumable Hugging Face download
/// manager, on-device embeddings + a SQLite vector store, device/arch detection,
/// and phone -> Wear OS model transfer over the Wearable Data Layer.
library;

// Inference runtime (isolate-backed FFI) and the Nishachar agent
// (part of the runtime library).
export 'src/runtime/sipllm_runtime.dart';
export 'src/runtime/sipllm_types.dart';

// Device / accelerator / architecture.
export 'src/device/sipllm_device.dart';
export 'src/device/arch.dart';

// Hugging Face download manager.
export 'src/download/hf_download_manager.dart';
export 'src/download/download_task.dart' show DownloadTask;

// On-device embeddings + vector store.
export 'src/embedding/embedding_store.dart';

// Phone <-> Wear OS model transfer.
export 'src/wear/wear_transfer.dart';
export 'src/wear/model_transfer.dart';
