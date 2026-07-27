// download_task.dart — the per-file state machine driving a resumable,
// multi-connection download. A task splits a ranged resource into contiguous
// byte segments, streams each into `<dest>.partN`, persists committed offsets
// to a `<dest>.sipdl.json` sidecar, and concatenates + verifies the parts into
// `<dest>` on completion. Pure `dart:io`; no Flutter imports.
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:meta/meta.dart';

import 'hf_download_manager.dart';

/// One contiguous byte range `[start, end]` (inclusive) of the target file,
/// streamed into its own part file. [written] is the number of bytes already
/// committed for this segment — the resume cursor.
class _Segment {
  _Segment({
    required this.index,
    required this.start,
    required this.end,
    this.written = 0,
  });

  factory _Segment.fromJson(Map<String, dynamic> json) => _Segment(
    index: json['index'] as int,
    start: json['start'] as int,
    end: json['end'] as int,
    written: json['written'] as int,
  );

  final int index;
  final int start;
  int end;
  int written;

  StreamSubscription<List<int>>? sub;
  Completer<void>? completer;

  int get length => end - start + 1;
  bool get complete => written >= length;

  Map<String, dynamic> toJson() => <String, dynamic>{
    'index': index,
    'start': start,
    'end': end,
    'written': written,
  };
}

class _RateSample {
  const _RateSample(this.ms, this.received);
  final int ms;
  final int received;
}

/// A single resumable download. Obtain one from [DownloadManager.enqueue];
/// observe [progress], await [done], and steer with [pause]/[resume]/[cancel].
class DownloadTask {
  @internal
  DownloadTask({
    required HttpClient client,
    required Future<void> Function() acquireConnection,
    required void Function() releaseConnection,
    required this.url,
    required this.destPath,
    required int connections,
    this.expectedSize,
    this.sha256Hex,
    Map<String, String>? headers,
  }) : _client = client,
       _acquire = acquireConnection,
       _release = releaseConnection,
       _connections = connections,
       _headers = headers;

  /// Source URL (Hugging Face `resolve` links and their CDN redirects welcome).
  final String url;

  /// Final assembled output path.
  final String destPath;

  /// Optional expected byte count, verified after assembly.
  final int? expectedSize;

  /// Optional lowercase-hex sha256, verified after assembly.
  final String? sha256Hex;

  final HttpClient _client;
  final Future<void> Function() _acquire;
  final void Function() _release;
  final int _connections;
  final Map<String, String>? _headers;

  final StreamController<DownloadProgress> _progress =
      StreamController<DownloadProgress>.broadcast();
  final Completer<void> _done = Completer<void>();
  final List<_RateSample> _rateSamples = <_RateSample>[];

  List<_Segment> _segments = <_Segment>[];
  DownloadState _state = DownloadState.queued;
  bool _probed = false;
  bool _rangeSupported = false;
  int? _total;
  int _received = 0;
  int _active = 0;
  String _resolvedUrl = '';

  String get targetUrl => _resolvedUrl.isNotEmpty ? _resolvedUrl : url;

  bool _stopRequested = false;
  bool _cancelRequested = false;
  Future<void>? _current;

  DateTime _lastEmit = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastCommit = DateTime.fromMillisecondsSinceEpoch(0);
  int _lastCommitBytes = 0;

  String get _sidecarPath => '$destPath.sipdl.json';
  String _partPath(int index) => '$destPath.part$index';

  /// Broadcast stream of throttled (~4/sec) progress samples. A terminal sample
  /// is emitted immediately before [done] settles.
  Stream<DownloadProgress> get progress => _progress.stream;

  /// Completes when the task reaches [DownloadState.completed] or
  /// [DownloadState.canceled]; completes with an error on
  /// [DownloadState.failed].
  Future<void> get done => _done.future;

  /// Current lifecycle state.
  DownloadState get state => _state;

  /// Resolves when the current run cycle has fully unwound — after a pause has
  /// flushed its sidecar, or after a terminal transition. Used by
  /// [DownloadManager.disposeAll].
  Future<void> get settled => _current ?? Future<void>.value();

  // --- Public controls ------------------------------------------------------

  /// Kicks off the first run. No-op once past [DownloadState.queued].
  void start() {
    if (_state != DownloadState.queued) return;
    _setState(DownloadState.running);
    _current = _runLoop();
  }

  /// Stops all in-flight connections promptly and flushes committed offsets to
  /// the sidecar so the task can be resumed later (even by a fresh process).
  void pause() {
    if (_state != DownloadState.running) return;
    _setState(DownloadState.paused);
    _stopRequested = true;
    _abortActive();
  }

  /// Restarts a paused (or failed) task from its committed offsets. Returns
  /// when the resumed run cycle ends.
  Future<void> resume() async {
    if (_state != DownloadState.paused && _state != DownloadState.failed) {
      return;
    }
    _setState(DownloadState.running);
    _stopRequested = false;
    _cancelRequested = false;
    _current = _runLoop();
    await _current;
  }

  /// Aborts the download and deletes every part file and the sidecar, leaving
  /// no trace. Transitions to [DownloadState.canceled].
  void cancel() {
    if (_state == DownloadState.completed || _state == DownloadState.canceled) {
      return;
    }
    _cancelRequested = true;
    _stopRequested = true;
    _setState(DownloadState.canceled);
    _abortActive();
    unawaited(_finishCancel());
  }

  // --- Core run loop ---------------------------------------------------------

  Future<void> _runLoop() async {
    try {
      if (!_probed) {
        await _probe();
        _probed = true;
      }
      if (_segments.isEmpty) {
        await _prepareSegments();
      }
      _emit(force: true);

      if (!_rangeSupported) {
        await _runSingle();
      } else {
        final pending = _segments.where((s) => !s.complete).map(_runSegment);
        await Future.wait(pending);
      }

      if (_stopRequested) {
        // Paused (or cancel is unwinding): persist offsets and rest. `done`
        // stays open for a pause; `_finishCancel` handles the cancel path.
        await _writeSidecar();
        _emit(force: true);
        return;
      }

      await _assemble();
      await _verify();
      await _cleanup(keepDest: true);
      _setState(DownloadState.completed);
      _received = _total ?? _received;
      _emit(force: true);
      await _progress.close();
      if (!_done.isCompleted) _done.complete();
    } catch (error, stack) {
      if (_cancelRequested) {
        // Cancellation surfaced as a socket error; `_finishCancel` owns cleanup.
        return;
      }
      _setState(DownloadState.failed);
      await _cleanup(keepDest: false);
      _emit(force: true);
      await _progress.close();
      if (!_done.isCompleted) _done.completeError(error, stack);
    }
  }

  Future<void> _finishCancel() async {
    try {
      await (_current ?? Future<void>.value());
    } catch (_) {
      // Errors during teardown are irrelevant once canceled.
    }
    await _cleanup(keepDest: false);
    _emit(force: true);
    if (!_progress.isClosed) await _progress.close();
    if (!_done.isCompleted) _done.complete();
  }

  // --- Probing ---------------------------------------------------------------

  Future<void> _probe() async {
    _resolvedUrl = await _resolveDirectUrl(url);
    final reqUrl = targetUrl;

    // Prefer a bodyless HEAD; most CDNs (and our tests) disclose Content-Length
    // and Accept-Ranges without shipping a byte.
    try {
      final request = await _client.openUrl('HEAD', Uri.parse(reqUrl));
      _applyHeaders(request);
      request.followRedirects = true;
      final response = await request.close();
      await response.drain<void>();
      if (response.statusCode >= 200 && response.statusCode < 300) {
        final length = response.contentLength;
        if (length >= 0) _total = length;
        final acceptRanges = response.headers.value(
          HttpHeaders.acceptRangesHeader,
        );
        _rangeSupported =
            acceptRanges != null &&
            acceptRanges.toLowerCase().contains('bytes') &&
            _total != null &&
            _total! > 0;
        if (_rangeSupported || _total != null) return;
      }
    } catch (_) {
      // Fall through to the GET probe below.
    }

    // Fallback: a one-byte ranged GET reveals both support and total size.
    final request = await _client.openUrl('GET', Uri.parse(reqUrl));
    _applyHeaders(request);
    request.followRedirects = true;
    request.headers.set(HttpHeaders.rangeHeader, 'bytes=0-0');
    final response = await request.close();
    if (response.statusCode == HttpStatus.partialContent) {
      final contentRange = response.headers.value(
        HttpHeaders.contentRangeHeader,
      );
      if (contentRange != null) {
        final slash = contentRange.indexOf('/');
        if (slash >= 0) {
          final total = int.tryParse(contentRange.substring(slash + 1).trim());
          if (total != null) _total = total;
        }
      }
      _rangeSupported = _total != null && _total! > 0;
    } else {
      _rangeSupported = false;
      if (response.contentLength >= 0) _total = response.contentLength;
    }
    await response.drain<void>();
  }

  Future<String> _resolveDirectUrl(String initialUrl) async {
    var current = initialUrl;
    for (var i = 0; i < 5; i++) {
      try {
        final req = await _client.openUrl('HEAD', Uri.parse(current));
        _applyHeaders(req);
        req.followRedirects = false;
        final res = await req.close();
        await res.drain<void>();
        if (res.statusCode == HttpStatus.movedPermanently ||
            res.statusCode == HttpStatus.found ||
            res.statusCode == HttpStatus.seeOther ||
            res.statusCode == HttpStatus.temporaryRedirect ||
            res.statusCode == 308) {
          final loc = res.headers.value(HttpHeaders.locationHeader);
          if (loc != null && loc.isNotEmpty) {
            current = Uri.parse(current).resolve(loc).toString();
            continue;
          }
        }
      } catch (_) {}
      break;
    }
    return current;
  }

  // --- Segment preparation (fresh vs resume) ---------------------------------

  Future<void> _prepareSegments() async {
    await File(destPath).parent.create(recursive: true);

    final sidecar = File(_sidecarPath);
    if (await sidecar.exists()) {
      try {
        final json = jsonDecode(await sidecar.readAsString());
        if (json is Map<String, dynamic> &&
            json['url'] == url &&
            json['rangeSupported'] == _rangeSupported &&
            json['total'] == _total) {
          final raw = (json['segments'] as List).cast<Map<String, dynamic>>();
          _segments = raw.map(_Segment.fromJson).toList();
          for (final segment in _segments) {
            await _truncatePartTo(segment.index, segment.written);
          }
          _received = _segments.fold<int>(0, (sum, s) => sum + s.written);
          return;
        }
      } catch (_) {
        // Corrupt or mismatched sidecar: fall through and start fresh.
      }
    }

    if (_rangeSupported && _total != null) {
      _segments = _split(_total!, _connections);
    } else {
      _segments = <_Segment>[
        _Segment(index: 0, start: 0, end: (_total ?? 0) - 1),
      ];
    }
    for (final segment in _segments) {
      await _truncatePartTo(segment.index, 0);
    }
    _received = 0;
    await _writeSidecar();
  }

  List<_Segment> _split(int total, int connections) {
    final count = connections < 1 ? 1 : connections;
    final per = total ~/ count;
    final segments = <_Segment>[];
    var start = 0;
    for (var i = 0; i < count; i++) {
      final end = (i == count - 1) ? total - 1 : start + per - 1;
      if (start > end) break; // More connections requested than bytes.
      segments.add(_Segment(index: segments.length, start: start, end: end));
      start = end + 1;
    }
    if (segments.isEmpty) {
      segments.add(_Segment(index: 0, start: 0, end: total - 1));
    }
    return segments;
  }

  // --- Segment download (multi-connection, resumable) ------------------------

  Future<void> _runSegment(_Segment segment) async {
    await _acquire();
    _active++;
    RandomAccessFile? raf;
    final completer = Completer<void>();
    segment.completer = completer;
    try {
      if (_stopRequested) {
        if (!completer.isCompleted) completer.complete();
        return;
      }
      final from = segment.start + segment.written;
      final to = segment.end;
      if (from > to) {
        if (!completer.isCompleted) completer.complete();
        return; // Already fully committed.
      }

      raf = await _openPartAppend(segment.index, segment.written);
      final request = await _client.openUrl('GET', Uri.parse(targetUrl));
      _applyHeaders(request);
      request.headers.set(HttpHeaders.rangeHeader, 'bytes=$from-$to');
      request.followRedirects = true;
      final response = await request.close();
      if (response.statusCode != HttpStatus.partialContent &&
          response.statusCode != HttpStatus.ok) {
        throw HttpException(
          'Unexpected status ${response.statusCode} for range $from-$to',
          uri: Uri.parse(targetUrl),
        );
      }

      final file = raf;
      late StreamSubscription<List<int>> sub;
      sub = response.listen(
        (chunk) async {
          sub.pause();
          try {
            await file.writeFrom(chunk);
            segment.written += chunk.length;
            _received += chunk.length;
            await _maybeCommit();
            _emit();
            if (_stopRequested) {
              await sub.cancel();
              if (!completer.isCompleted) completer.complete();
              return;
            }
            sub.resume();
          } catch (error, stack) {
            await sub.cancel();
            if (!completer.isCompleted) completer.completeError(error, stack);
          }
        },
        onDone: () {
          if (!completer.isCompleted) completer.complete();
        },
        onError: (Object error, StackTrace stack) {
          if (!completer.isCompleted) completer.completeError(error, stack);
        },
        cancelOnError: true,
      );
      segment.sub = sub;
      await completer.future;
    } finally {
      segment.sub = null;
      if (raf != null) {
        try {
          await raf.flush();
        } catch (_) {}
        try {
          await raf.close();
        } catch (_) {}
      }
      _active--;
      _release();
    }
  }

  // --- Single-stream fallback (no range support, non-resumable) --------------

  Future<void> _runSingle() async {
    await _acquire();
    _active++;
    final segment = _segments[0];
    RandomAccessFile? raf;
    final completer = Completer<void>();
    segment.completer = completer;
    try {
      // No range support means no resume: always restart from zero.
      segment.written = 0;
      _received = 0;
      raf = await _openPartTruncate(segment.index);
      final request = await _client.openUrl('GET', Uri.parse(targetUrl));
      _applyHeaders(request);
      request.followRedirects = true;
      final response = await request.close();
      if (response.statusCode != HttpStatus.ok &&
          response.statusCode != HttpStatus.partialContent) {
        throw HttpException(
          'Unexpected status ${response.statusCode}',
          uri: Uri.parse(targetUrl),
        );
      }
      if (_total == null && response.contentLength >= 0) {
        _total = response.contentLength;
      }

      final file = raf;
      late StreamSubscription<List<int>> sub;
      sub = response.listen(
        (chunk) async {
          sub.pause();
          try {
            await file.writeFrom(chunk);
            segment.written += chunk.length;
            _received += chunk.length;
            _emit();
            if (_stopRequested) {
              await sub.cancel();
              if (!completer.isCompleted) completer.complete();
              return;
            }
            sub.resume();
          } catch (error, stack) {
            await sub.cancel();
            if (!completer.isCompleted) completer.completeError(error, stack);
          }
        },
        onDone: () {
          if (!completer.isCompleted) completer.complete();
        },
        onError: (Object error, StackTrace stack) {
          if (!completer.isCompleted) completer.completeError(error, stack);
        },
        cancelOnError: true,
      );
      segment.sub = sub;
      await completer.future;
      if (!_stopRequested) {
        // Pin the real end so assembly/verification know the exact size.
        segment.end = segment.written - 1;
      }
    } finally {
      segment.sub = null;
      if (raf != null) {
        try {
          await raf.flush();
        } catch (_) {}
        try {
          await raf.close();
        } catch (_) {}
      }
      _active--;
      _release();
    }
  }

  // --- Assembly / verification / cleanup -------------------------------------

  Future<void> _assemble() async {
    final out = File(destPath);
    await out.parent.create(recursive: true);
    final sink = out.openWrite(mode: FileMode.write);
    try {
      for (final segment in _segments) {
        await sink.addStream(File(_partPath(segment.index)).openRead());
      }
    } finally {
      await sink.close();
    }
  }

  Future<void> _verify() async {
    final length = await File(destPath).length();
    if (_total != null && _rangeSupported && length != _total) {
      throw StateError('Size mismatch: expected $_total, got $length');
    }
    if (expectedSize != null && length != expectedSize) {
      throw StateError('Size mismatch: expected $expectedSize, got $length');
    }
    if (sha256Hex != null) {
      final actual = await _fileSha256(destPath);
      if (actual.toLowerCase() != sha256Hex!.toLowerCase()) {
        throw StateError('sha256 mismatch: expected $sha256Hex, got $actual');
      }
    }
  }

  Future<String> _fileSha256(String path) async {
    Digest? digest;
    final sink = sha256.startChunkedConversion(
      ChunkedConversionSink<Digest>.withCallback((digests) {
        digest = digests.single;
      }),
    );
    await for (final chunk in File(path).openRead()) {
      sink.add(chunk);
    }
    sink.close();
    return digest!.toString();
  }

  Future<void> _cleanup({required bool keepDest}) async {
    for (final segment in _segments) {
      await _deleteIfExists(_partPath(segment.index));
    }
    await _deleteIfExists(_sidecarPath);
    if (!keepDest) {
      await _deleteIfExists(destPath);
    }
  }

  // --- Progress + sidecar persistence ----------------------------------------

  Future<void> _maybeCommit() async {
    final now = DateTime.now();
    final elapsed = now.difference(_lastCommit).inMilliseconds;
    if (elapsed < 250 && (_received - _lastCommitBytes) < (1 << 20)) return;
    _lastCommit = now;
    _lastCommitBytes = _received;
    await _writeSidecar();
  }

  Future<void> _writeSidecar() async {
    final payload = jsonEncode(<String, dynamic>{
      'url': url,
      'total': _total,
      'sha256': sha256Hex,
      'rangeSupported': _rangeSupported,
      'connections': _connections,
      'segments': _segments.map((s) => s.toJson()).toList(),
    });
    final tmp = File('$_sidecarPath.tmp');
    await tmp.writeAsString(payload, flush: true);
    await _safeRename(tmp, _sidecarPath);
  }

  void _emit({bool force = false}) {
    final now = DateTime.now();
    if (!force && now.difference(_lastEmit).inMilliseconds < 250) return;
    _lastEmit = now;

    final nowMs = now.millisecondsSinceEpoch;
    _rateSamples.add(_RateSample(nowMs, _received));
    while (_rateSamples.length > 2 && nowMs - _rateSamples.first.ms > 2000) {
      _rateSamples.removeAt(0);
    }
    var bytesPerSecond = 0.0;
    if (_rateSamples.length >= 2) {
      final first = _rateSamples.first;
      final last = _rateSamples.last;
      final dt = (last.ms - first.ms) / 1000.0;
      if (dt > 0) bytesPerSecond = (last.received - first.received) / dt;
    }
    if (!_progress.isClosed) {
      _progress.add(
        DownloadProgress(
          received: _received,
          total: _total,
          bytesPerSecond: bytesPerSecond,
          activeConnections: _active,
        ),
      );
    }
  }

  // --- Small helpers ---------------------------------------------------------

  void _abortActive() {
    for (final segment in _segments) {
      final sub = segment.sub;
      segment.sub = null;
      if (sub != null) unawaited(sub.cancel());
      final completer = segment.completer;
      if (completer != null && !completer.isCompleted) completer.complete();
    }
  }

  void _applyHeaders(HttpClientRequest request) {
    // Identity encoding keeps byte offsets meaningful for ranged requests.
    request.headers.set(HttpHeaders.acceptEncodingHeader, 'identity');
    _headers?.forEach(request.headers.set);
  }

  void _setState(DownloadState next) => _state = next;

  Future<RandomAccessFile> _openPartAppend(int index, int written) async {
    final file = File(_partPath(index));
    await file.parent.create(recursive: true);
    if (!await file.exists()) {
      await file.create();
    }
    final raf = await file.open(mode: FileMode.append);
    // Drop any bytes past the committed offset, then append continues at EOF.
    await raf.truncate(written);
    return raf;
  }

  Future<RandomAccessFile> _openPartTruncate(int index) async {
    final file = File(_partPath(index));
    await file.parent.create(recursive: true);
    return file.open(mode: FileMode.write);
  }

  Future<void> _truncatePartTo(int index, int length) async {
    final file = File(_partPath(index));
    if (!await file.exists()) {
      await file.parent.create(recursive: true);
      await file.create();
    }
    final raf = await file.open(mode: FileMode.append);
    try {
      await raf.truncate(length);
    } finally {
      await raf.close();
    }
  }

  Future<void> _safeRename(File source, String destPath) async {
    try {
      await source.rename(destPath);
    } catch (_) {
      try {
        await source.copy(destPath);
        await source.delete();
      } catch (_) {}
    }
  }

  Future<void> _deleteIfExists(String path) async {
    final file = File(path);
    if (await file.exists()) {
      try {
        await file.delete();
      } catch (_) {}
    }
  }
}
