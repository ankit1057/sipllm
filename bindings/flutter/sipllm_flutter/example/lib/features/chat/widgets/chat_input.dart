import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';

import '../../../core/theme/app_theme.dart';

/// The bottom composer: a multiline text field plus a Send / Stop action.
///
/// When [generating] is true the send affordance becomes a red Stop button
/// wired to [onStop]; otherwise it sends the trimmed field contents via
/// [onSend]. The whole row is disabled when [enabled] is false (no model).
class ChatInput extends StatefulWidget {
  final bool enabled;
  final bool generating;
  final ValueChanged<String> onSend;
  final VoidCallback onStop;

  const ChatInput({
    super.key,
    required this.enabled,
    required this.generating,
    required this.onSend,
    required this.onStop,
  });

  @override
  State<ChatInput> createState() => _ChatInputState();
}

class _ChatInputState extends State<ChatInput> {
  final _controller = TextEditingController();
  final _focus = FocusNode();

  @override
  void dispose() {
    _controller.dispose();
    _focus.dispose();
    super.dispose();
  }

  void _submit() {
    final text = _controller.text.trim();
    if (text.isEmpty || !widget.enabled || widget.generating) return;
    widget.onSend(text);
    _controller.clear();
    _focus.requestFocus();
  }

  @override
  Widget build(BuildContext context) {
    final generating = widget.generating;
    return SafeArea(
      top: false,
      child: Container(
        padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
        decoration: BoxDecoration(
          color: AppTheme.background,
          border: Border(
            top: BorderSide(color: Colors.white.withValues(alpha: 0.05)),
          ),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Expanded(
              child: TextField(
                controller: _controller,
                focusNode: _focus,
                enabled: widget.enabled && !generating,
                minLines: 1,
                maxLines: 5,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => _submit(),
                style: const TextStyle(color: AppTheme.textMain),
                decoration: InputDecoration(
                  hintText: widget.enabled ? 'Message…' : 'Load a model to chat',
                  contentPadding:
                      const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                ),
              ),
            ),
            const SizedBox(width: 8),
            _actionButton(generating),
          ],
        ),
      ),
    );
  }

  Widget _actionButton(bool generating) {
    if (generating) {
      return _circle(
        color: Colors.redAccent,
        icon: LucideIcons.square,
        onTap: widget.onStop,
      );
    }
    final active = widget.enabled;
    return _circle(
      color: active ? AppTheme.primary : AppTheme.surface,
      icon: LucideIcons.send,
      iconColor: active ? Colors.black : AppTheme.textSec,
      onTap: active ? _submit : null,
    );
  }

  Widget _circle({
    required Color color,
    required IconData icon,
    Color iconColor = Colors.white,
    VoidCallback? onTap,
  }) {
    return Material(
      color: color,
      shape: const CircleBorder(),
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Icon(icon, size: 20, color: iconColor),
        ),
      ),
    );
  }
}
