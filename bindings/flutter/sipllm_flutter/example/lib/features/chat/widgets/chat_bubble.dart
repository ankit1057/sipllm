import 'package:flutter/material.dart';
import '../../../core/models/chat_message.dart';
import '../../../core/theme/app_theme.dart';

class ChatBubble extends StatelessWidget {
  final ChatMessage message;

  const ChatBubble({super.key, required this.message});

  String _formatTime(DateTime dt) {
    final hour = dt.hour.toString().padLeft(2, '0');
    final minute = dt.minute.toString().padLeft(2, '0');
    return '$hour:$minute';
  }

  @override
  Widget build(BuildContext context) {
    final isUser = message.isUser;
    final timeString = _formatTime(message.timestamp);

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
      child: Row(
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (!isUser) ...[
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: AppTheme.surface,
                shape: BoxShape.circle,
                border: Border.all(color: AppTheme.primary.withValues(alpha: 0.3)),
              ),
              child: const Icon(Icons.smart_toy_outlined, size: 18, color: AppTheme.primary),
            ),
            const SizedBox(width: 8),
          ],

          Flexible(
            child: Container(
              padding: const EdgeInsets.only(left: 14, right: 14, top: 12, bottom: 8),
              constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.75),
              decoration: BoxDecoration(
                color: isUser ? AppTheme.primary.withValues(alpha: 0.15) : AppTheme.surface,
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(18),
                  topRight: const Radius.circular(18),
                  bottomLeft: isUser ? const Radius.circular(18) : Radius.zero,
                  bottomRight: isUser ? Radius.zero : const Radius.circular(18),
                ),
                border: Border.all(
                  color: isUser ? AppTheme.primary.withValues(alpha: 0.3) : Colors.white.withValues(alpha: 0.05),
                ),
              ),
              child: IntrinsicWidth(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Padding(
                      padding: const EdgeInsets.only(right: 8.0, bottom: 4.0),
                      child: Text(
                        message.text,
                        style: const TextStyle(
                          color: AppTheme.textMain,
                          fontSize: 15,
                          height: 1.4,
                        ),
                      ),
                    ),

                    Align(
                      alignment: Alignment.bottomRight,
                      child: Text(
                        timeString,
                        style: TextStyle(
                          fontSize: 10,
                          color: AppTheme.textSec.withValues(alpha: 0.7),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          if (isUser) ...[
            const SizedBox(width: 8),
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: AppTheme.primary,
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.person, size: 18, color: Colors.black),
            ),
          ],
        ],
      ),
    );
  }
}