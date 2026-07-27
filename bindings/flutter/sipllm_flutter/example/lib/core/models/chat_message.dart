import 'package:sipllm_flutter/sipllm_flutter.dart';

/// A single chat turn. [text] is mutable so the assistant message can grow
/// token-by-token during streaming; [stats] is attached when a turn completes.
class ChatMessage {
  ChatMessage({
    required this.text,
    required this.isUser,
    this.streaming = false,
    this.stats,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  String text;
  final bool isUser;
  bool streaming;
  SipllmStats? stats;
  final DateTime timestamp;
}
