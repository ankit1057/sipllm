// utf8_pieces.dart — the engine's token pieces as text.
//
// A piece is the raw bytes of one token. Byte-level BPE and SentencePiece
// byte-fallback tokens can split one multi-byte UTF-8 character (an emoji, most
// non-Latin scripts) across two or more pieces, and a model can emit bytes that
// never form a character. Decoding each piece strictly threw inside the native
// token callback, whose exceptional return (0) stopped the generation, and an
// agent step then ended as "cancelled".
import 'dart:convert';

/// Decodes one generation's pieces in order. A character split across pieces
/// comes out whole with the piece that completes it (earlier pieces give '');
/// bytes that can never be a character become U+FFFD. Never throws.
class Utf8PieceDecoder {
  Utf8PieceDecoder() {
    _sink = const Utf8Decoder(allowMalformed: true)
        .startChunkedConversion(StringConversionSink.fromStringSink(_out));
  }

  final StringBuffer _out = StringBuffer();
  late final ByteConversionSink _sink;
  bool _closed = false;

  /// The text [bytes] completes, possibly ''.
  String add(List<int> bytes) {
    if (_closed) return '';
    _sink.add(bytes);
    return _take();
  }

  /// Ends the generation: U+FFFD for a character left unfinished, else ''.
  String close() {
    if (_closed) return '';
    _closed = true;
    _sink.close();
    return _take();
  }

  String _take() {
    if (_out.isEmpty) return '';
    final s = _out.toString();
    _out.clear();
    return s;
  }
}
