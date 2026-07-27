import '../models/chat_message.dart';

/// Chat prompt templates. SipLLM does inference over raw text (it does not
/// auto-apply a chat template), so the app formats the conversation itself.
/// Special tokens are emitted as text; the byte-level BPE tokenizer encodes
/// them literally, which the instruct models still follow well in practice.
enum PromptTemplate {
  /// ChatML — SmolLM2, Qwen2/2.5.
  chatml,

  /// Llama-3.x instruct header format.
  llama3,

  /// TinyLlama / Zephyr chat.
  zephyr,

  /// Plain completion (no template) — for base models / raw mode.
  raw;

  /// Pick a template from a model id/name.
  static PromptTemplate forModel(String modelId) {
    final m = modelId.toLowerCase();
    if (m.contains('llama3') || m.contains('llama-3') || m.contains('llama3.')) {
      return PromptTemplate.llama3;
    }
    if (m.contains('tinyllama')) return PromptTemplate.zephyr;
    if (m.contains('qwen') || m.contains('smollm')) return PromptTemplate.chatml;
    return PromptTemplate.chatml;
  }

  /// Build the full prompt for a conversation, ending at the assistant turn.
  String build(String system, List<ChatMessage> history) {
    switch (this) {
      case PromptTemplate.chatml:
        final b = StringBuffer();
        if (system.isNotEmpty) {
          b.write('<|im_start|>system\n$system<|im_end|>\n');
        }
        for (final m in history) {
          final role = m.isUser ? 'user' : 'assistant';
          b.write('<|im_start|>$role\n${m.text}<|im_end|>\n');
        }
        b.write('<|im_start|>assistant\n');
        return b.toString();

      case PromptTemplate.llama3:
        final b = StringBuffer('<|begin_of_text|>');
        if (system.isNotEmpty) {
          b.write('<|start_header_id|>system<|end_header_id|>\n\n$system<|eot_id|>');
        }
        for (final m in history) {
          final role = m.isUser ? 'user' : 'assistant';
          b.write('<|start_header_id|>$role<|end_header_id|>\n\n${m.text}<|eot_id|>');
        }
        b.write('<|start_header_id|>assistant<|end_header_id|>\n\n');
        return b.toString();

      case PromptTemplate.zephyr:
        final b = StringBuffer();
        if (system.isNotEmpty) b.write('<|system|>\n$system</s>\n');
        for (final m in history) {
          final role = m.isUser ? 'user' : 'assistant';
          b.write('<|$role|>\n${m.text}</s>\n');
        }
        b.write('<|assistant|>\n');
        return b.toString();

      case PromptTemplate.raw:
        return history.isEmpty ? '' : history.last.text;
    }
  }
}
