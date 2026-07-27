import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  static const Color background = Color(0xFF0F1115);
  static const Color surface = Color(0xFF1E2028);
  static const Color primary = Color(0xFF00E676);
  static const Color accent = Color(0xFF6C63FF);
  static const Color textMain = Colors.white;
  static const Color textSec = Color(0xFF9CA3AF);

  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: background,
      primaryColor: primary,

      textTheme: GoogleFonts.spaceMonoTextTheme(
        ThemeData.dark().textTheme,
      ),

      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surface,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: Colors.white.withValues(alpha: 0.05)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: primary),
        ),
        hintStyle: const TextStyle(color: textSec),
      ),

      cardColor: surface,
      navigationBarTheme: NavigationBarThemeData(
        labelTextStyle: WidgetStateProperty.all(
          const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, overflow: TextOverflow.ellipsis),
        ),
      ),
    );
  }
}