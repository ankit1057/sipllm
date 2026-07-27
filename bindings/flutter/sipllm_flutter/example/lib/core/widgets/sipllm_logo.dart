import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

/// Custom Vector Logo Art for SipLLM Studio:
/// Combines a futuristic glowing cup silhouette, streaming neural data nodes,
/// and a central lightning processing core.
class SipLlmLogo extends StatefulWidget {
  const SipLlmLogo({
    super.key,
    this.size = 48.0,
    this.animated = true,
  });

  final double size;
  final bool animated;

  @override
  State<SipLlmLogo> createState() => _SipLlmLogoState();
}

class _SipLlmLogoState extends State<SipLlmLogo> with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 4),
    );
    if (widget.animated) {
      _controller.repeat();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return CustomPaint(
          size: Size(widget.size, widget.size),
          painter: _SipLlmLogoPainter(progress: _controller.value),
        );
      },
    );
  }
}

class _SipLlmLogoPainter extends CustomPainter {
  _SipLlmLogoPainter({required this.progress});

  final double progress;

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;
    final center = Offset(w / 2, h / 2);

    // 1. Outer Glow Aura Ring
    final auraPaint = Paint()
      ..shader = SweepGradient(
        colors: const [
          AppTheme.primary,
          AppTheme.accent,
          Colors.cyanAccent,
          AppTheme.primary,
        ],
        transform: GradientRotation(progress * 2 * math.pi),
      ).createShader(Rect.fromCircle(center: center, radius: w * 0.42))
      ..style = PaintingStyle.stroke
      ..strokeWidth = w * 0.05
      ..maskFilter = MaskFilter.blur(BlurStyle.solid, w * 0.08);

    canvas.drawCircle(center, w * 0.42, auraPaint);

    // 2. Futuristic Cup / Flask Vessel Path
    final vesselPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          AppTheme.primary.withValues(alpha: 0.9),
          AppTheme.accent.withValues(alpha: 0.9),
        ],
      ).createShader(Rect.fromLTWH(0, 0, w, h))
      ..style = PaintingStyle.fill;

    final path = Path();
    // Top Rim
    path.moveTo(w * 0.25, h * 0.22);
    path.quadraticBezierTo(w * 0.5, h * 0.18, w * 0.75, h * 0.22);
    // Body tapering down
    path.cubicTo(w * 0.82, h * 0.5, w * 0.65, h * 0.78, w * 0.5, h * 0.84);
    path.cubicTo(w * 0.35, h * 0.78, w * 0.18, h * 0.5, w * 0.25, h * 0.22);
    path.close();

    canvas.drawPath(path, vesselPaint);

    // 3. Inner Neural Lightning Stream
    final corePaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = w * 0.06
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final bolt = Path();
    bolt.moveTo(w * 0.52, h * 0.28);
    bolt.lineTo(w * 0.42, h * 0.46);
    bolt.lineTo(w * 0.56, h * 0.48);
    bolt.lineTo(w * 0.46, h * 0.70);

    canvas.drawPath(bolt, corePaint);

    // 4. Neural Nodes (Pulsing Orbs)
    final nodePaint = Paint()
      ..color = Colors.cyanAccent
      ..style = PaintingStyle.fill;

    final nodes = [
      Offset(w * 0.3, h * 0.32),
      Offset(w * 0.7, h * 0.32),
      Offset(w * 0.5, h * 0.78),
    ];

    for (int i = 0; i < nodes.length; i++) {
      final pulse = math.sin((progress * 2 * math.pi) + i) * 0.2 + 1.0;
      canvas.drawCircle(nodes[i], w * 0.04 * pulse, nodePaint);
    }
  }

  @override
  bool shouldRepaint(covariant _SipLlmLogoPainter oldDelegate) {
    return oldDelegate.progress != progress;
  }
}
