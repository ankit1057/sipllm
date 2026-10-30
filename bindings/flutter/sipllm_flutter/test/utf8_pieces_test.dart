// utf8_pieces_test.dart — token pieces that split or break UTF-8 must decode
// without throwing (a throw in the native token callback stopped generation).
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:sipllm_flutter/src/runtime/utf8_pieces.dart';

void main() {
  test('an emoji split over byte pieces comes out whole on its last piece', () {
    final d = Utf8PieceDecoder();
    final bytes = utf8.encode('🙂'); // F0 9F 99 82
    expect(bytes, hasLength(4));
    final out = [for (final b in bytes) d.add([b])];
    expect(out, ['', '', '', '🙂']);
    expect(d.close(), '');
  });

  test('a character split across ordinary pieces joins up', () {
    final d = Utf8PieceDecoder();
    final hi = utf8.encode('नमस्ते'); // Devanagari, 3 bytes per code point
    final out = StringBuffer()
      ..write(d.add(utf8.encode('Hello ')))
      ..write(d.add(hi.sublist(0, 4)))
      ..write(d.add(hi.sublist(4)))
      ..write(d.add(utf8.encode('!')));
    expect(out.toString(), 'Hello नमस्ते!');
  });

  test('bytes that are never a character become U+FFFD instead of throwing', () {
    final d = Utf8PieceDecoder();
    expect(() => utf8.decode([0xFF]), throwsFormatException); // what toDartString() did
    expect(d.add([0x61, 0xFF, 0x62]), 'a�b');
    expect(d.add([0xC3]), ''); // starts a 2-byte character
    expect(d.add([0x41]), '�A'); // ...that never arrives
  });

  test('an unfinished character at the end is reported by close()', () {
    final d = Utf8PieceDecoder();
    expect(d.add([0x6F, 0x6B, 0xE2, 0x82]), 'ok');
    expect(d.close(), '�');
    expect(d.add([0x61]), '', reason: 'closed');
  });
}
