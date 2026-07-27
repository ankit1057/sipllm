// hf_download_manager.dart — a resumable, multi-connection downloader built on
// `dart:io` `HttpClient` alone (no Flutter, no `package:http`), so it can be
// unit-tested on the host VM under `flutter test`. Purpose-built for pulling
// large model weights off Hugging Face onto phones and watches where a dropped
// connection or a killed process must never mean starting over.
import 'dart:async';
import 'dart:collection';
import 'dart:io';

import 'download_task.dart';

/// Points at a file inside a Hugging Face repository revision and builds the
/// `resolve` URL the CDN understands. The resolve URL 302-redirects to a signed
/// CDN location; [DownloadManager] follows that automatically.
class HuggingFaceRepo {
  const HuggingFaceRepo(this.repo, {this.revision = 'main'});

  /// `<owner>/<name>`, e.g. `Qwen/Qwen2.5-0.5B-Instruct-GGUF`.
  final String repo;

  /// Git revision — a branch, tag, or commit sha. Defaults to `main`.
  final String revision;

  /// `https://huggingface.co/<repo>/resolve/<revision>/<filename>`.
  String resolveUrl(String filename) =>
      'https://huggingface.co/$repo/resolve/$revision/$filename';
}

/// Lifecycle of a single [DownloadTask]. Only [completed], [failed] and
/// [canceled] are terminal (they complete [DownloadTask.done]); [paused] is a
/// resting state that [DownloadTask.resume] revives.
enum DownloadState { queued, running, paused, completed, failed, canceled }

/// An immutable progress sample emitted on [DownloadTask.progress].
class DownloadProgress {
  const DownloadProgress({
    required this.received,
    required this.total,
    required this.bytesPerSecond,
    required this.activeConnections,
  });

  /// Bytes committed to disk so far across every segment.
  final int received;

  /// Total size in bytes, or `null` when the server never disclosed a length
  /// (a non-range, chunked stream).
  final int? total;

  /// Moving-average throughput over roughly the last two seconds.
  final double bytesPerSecond;

  /// Sockets actively streaming right now for this task.
  final int activeConnections;

  /// Completion in `[0, 1]`, or `null` when [total] is unknown.
  double? get fraction {
    final t = total;
    if (t == null || t <= 0) return null;
    final f = received / t;
    return f < 0 ? 0 : (f > 1 ? 1 : f);
  }

  @override
  String toString() =>
      'DownloadProgress(received: $received, total: $total, '
      'bps: ${bytesPerSecond.toStringAsFixed(0)}, conns: $activeConnections)';
}

/// Owns a shared [HttpClient] and a global connection budget, and mints
/// [DownloadTask]s. The [maxConnections] cap is enforced across *all* tasks:
/// individual tasks may request more segments, but only [maxConnections]
/// sockets ever stream simultaneously.
class DownloadManager {
  DownloadManager({int maxConnections = 4})
    : _maxConnections = maxConnections < 1 ? 1 : maxConnections {
    _client = HttpClient()
      // Byte-exact ranges require the wire bytes untouched: never let the
      // client transparently gunzip a segment, and ask origins for identity.
      ..autoUncompress = false
      ..idleTimeout = const Duration(minutes: 10)
      ..connectionTimeout = const Duration(seconds: 30)
      ..maxConnectionsPerHost = (maxConnections < 1 ? 1 : maxConnections) + 4;
    _permits = _maxConnections;
  }

  final int _maxConnections;
  late final HttpClient _client;
  late int _permits;
  final Queue<Completer<void>> _waiters = Queue<Completer<void>>();
  final List<DownloadTask> _tasks = <DownloadTask>[];

  /// All tasks ever enqueued on this manager, newest last.
  List<DownloadTask> get tasks => List<DownloadTask>.unmodifiable(_tasks);

  Future<void> _acquire() {
    if (_permits > 0) {
      _permits--;
      return Future<void>.value();
    }
    final completer = Completer<void>();
    _waiters.add(completer);
    return completer.future;
  }

  void _release() {
    if (_waiters.isNotEmpty) {
      _waiters.removeFirst().complete();
    } else {
      _permits++;
    }
  }

  /// Creates and starts a download of [url] into [destPath].
  ///
  /// If [destPath] already has a sidecar/part set from a prior run, it resumes
  /// from the committed offsets instead of restarting. [connections] overrides
  /// the desired segment count (defaults to [maxConnections]); [expectedSize]
  /// and [sha256] enable post-assembly verification; [headers] (e.g. an
  /// `Authorization` bearer for a gated repo) are applied to every request.
  DownloadTask enqueue({
    required String url,
    required String destPath,
    int? expectedSize,
    String? sha256,
    int? connections,
    Map<String, String>? headers,
  }) {
    final desired = connections ?? _maxConnections;
    final task = DownloadTask(
      client: _client,
      acquireConnection: _acquire,
      releaseConnection: _release,
      url: url,
      destPath: destPath,
      connections: desired < 1 ? 1 : desired,
      expectedSize: expectedSize,
      sha256Hex: sha256,
      headers: headers,
    );
    _tasks.add(task);
    task.start();
    return task;
  }

  /// Pauses every task and closes the shared client. In-flight tasks flush
  /// their offsets to their sidecars first, so a later [DownloadManager] can
  /// resume them. Safe to call once at app shutdown.
  Future<void> disposeAll() async {
    for (final task in _tasks) {
      task.pause();
    }
    await Future.wait(_tasks.map((t) => t.settled));
    _client.close(force: true);
  }
}
