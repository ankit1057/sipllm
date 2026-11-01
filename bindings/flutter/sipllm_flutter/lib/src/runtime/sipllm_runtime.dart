// sipllm_runtime.dart — the public, isolate-backed inference runtime.
//
// One [SipllmRuntime] owns one worker isolate holding one native context. It
// exposes streaming generation, cancellation, embeddings, and per-run stats.
import 'dart:async';
import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import '../ffi/native_library.dart';
import '../ffi/sipllm_bindings.dart';
import 'sipllm_isolate.dart';
import 'sipllm_types.dart';

part 'sipllm_agent.dart';

class SipllmException implements Exception {
  SipllmException(this.message);
  final String message;
  @override
  String toString() => 'SipllmException: $message';
}

/// A loaded SipLLM model, driven from a background isolate.
///
/// ```dart
/// final rt = await SipllmRuntime.open(
///   '/path/model.gguf',
///   params: const SipllmParams(ramBudgetBytes: 512 * 1024 * 1024, fastQuant: true),
/// );
/// await for (final tok in rt.generate('The capital of France is')) {
///   stdout.write(tok.piece);
/// }
/// print(rt.lastStats?.decodeTokensPerSecond);
/// await rt.close();
/// ```
class SipllmRuntime {
  SipllmRuntime._(
    this._isolate,
    this._toWorker,
    this._ctxAddress,
    this._cancelFn,
    this.modelInfo,
    this.threads,
    this._exitPort,
    Stream<Object?> exitEvents,
  ) {
    // A worker that dies (an uncaught error, a native abort that unwinds to
    // Dart) must not leave callers waiting on replies that will never come.
    exitEvents.listen((message) {
      _workerGone = true;
      _closed = true;
      _busy = false;
      final why = message is List && message.isNotEmpty
          ? 'sipllm worker failed: ${message.first}'
          : 'sipllm worker exited';
      for (final fail in _inflight.toList()) {
        fail(why);
      }
      _inflight.clear();
      _exitPort.close();
    });
  }

  final Isolate _isolate;
  final ReceivePort _exitPort;
  bool _workerGone = false;
  // One failer per in-flight request; called with a reason if the worker dies.
  final Set<void Function(String)> _inflight = {};
  final SendPort _toWorker;
  final int _ctxAddress;
  final void Function(Pointer<SipllmCtx>) _cancelFn;

  /// Static description of the loaded model.
  final SipllmModelInfo modelInfo;

  /// Worker thread count the engine actually spun up.
  final int threads;

  SipllmStats? _lastStats;
  bool _busy = false;
  bool _closed = false;

  /// Metrics from the most recent completed generation.
  SipllmStats? get lastStats => _lastStats;

  bool get isBusy => _busy;

  /// Open [modelPath] (GGUF or .llmw) on a worker isolate.
  static Future<SipllmRuntime> open(
    String modelPath, {
    SipllmParams params = const SipllmParams(),
    String? libraryPath,
    int logLevel = 1,
  }) async {
    // The main isolate loads its own handle so it can call the thread-safe
    // cancel() on the shared context while the worker is blocked in generate().
    final mainBindings = loadSipllmBindings(libraryPath: libraryPath);

    final init = ReceivePort();
    // Registered at spawn, so a worker that dies at any point is reported;
    // broadcast so the open handshake and the runtime can both listen.
    final exit = ReceivePort();
    final exitEvents = exit.asBroadcastStream();
    final isolate = await Isolate.spawn(
      sipllmWorkerMain,
      SipllmWorkerInit(
          init.sendPort, libraryPath, modelPath, params.toMap(), logLevel),
      errorsAreFatal: true,
      onError: exit.sendPort,
      onExit: exit.sendPort,
      debugName: 'sipllm-worker',
    );

    // The first message is the worker's hello, or its death notice if it
    // failed before saying hello.
    final firstEvent = Completer<Object?>();
    init.listen((m) {
      if (!firstEvent.isCompleted) firstEvent.complete(m);
    });
    // Left subscribed (it ignores everything after the handshake): cancelling
    // would await, and a death notice in that gap would reach no listener.
    exitEvents.listen((m) {
      if (!firstEvent.isCompleted) {
        firstEvent.complete({'error': 'sipllm worker exited during open: $m'});
      }
    });
    final first = await firstEvent.future as Map;
    init.close();
    if (first['error'] != null) {
      exit.close();
      isolate.kill(priority: Isolate.immediate);
      throw SipllmException(first['error'] as String);
    }
    return SipllmRuntime._(
      isolate,
      first['toWorker'] as SendPort,
      first['ctxAddress'] as int,
      mainBindings.cancel,
      SipllmModelInfo.fromMap(first['modelInfo'] as Map<String, Object?>),
      first['threads'] as int,
      exit,
      exitEvents,
    );
  }

  /// Stream generated pieces for [prompt]. Populates [lastStats] on completion.
  /// Throws if a generation is already in flight (a runtime is single-context).
  Stream<SipllmToken> generate(
    String prompt, {
    int maxNew = 256,
    SipllmSampler sampler = const SipllmSampler(),
  }) {
    if (_closed) {
      return Stream.error(SipllmException('runtime is closed'));
    }
    if (_busy) {
      return Stream.error(
          SipllmException('a generation is already in progress'));
    }
    _busy = true;
    final controller = StreamController<SipllmToken>();
    final port = ReceivePort();
    late final void Function(String) fail;
    fail = (why) {
      controller.addError(SipllmException(why));
      _finish(port, controller);
    };
    _inflight.add(fail);

    port.listen((message) {
      final m = message as Map;
      if (m['token'] != null) {
        controller.add(SipllmToken(m['token'] as String, m['id'] as int));
      } else if (m['done'] == true) {
        _inflight.remove(fail);
        _lastStats = SipllmStats.fromMap(m['stats'] as Map<String, Object?>);
        _finish(port, controller);
      } else if (m['error'] != null) {
        _inflight.remove(fail);
        controller.addError(SipllmException(m['error'] as String));
        _finish(port, controller);
      }
    });

    controller.onCancel = () {
      // Consumer stopped listening: ask the engine to stop too.
      if (_busy) cancel();
    };

    _toWorker.send({
      'cmd': 'generate',
      'prompt': prompt,
      'maxNew': maxNew,
      'sampler': sampler.toMap(),
      'port': port.sendPort,
    });
    return controller.stream;
  }

  void _finish(ReceivePort port, StreamController<SipllmToken> controller) {
    _busy = false;
    port.close();
    if (!controller.isClosed) controller.close();
  }

  /// Convenience: collect the full completion as a single string.
  Future<String> complete(
    String prompt, {
    int maxNew = 256,
    SipllmSampler sampler = const SipllmSampler(),
  }) async {
    final buf = StringBuffer();
    await for (final tok in generate(prompt, maxNew: maxNew, sampler: sampler)) {
      buf.write(tok.piece);
    }
    return buf.toString();
  }

  /// Ask the in-flight generation to stop at the next token boundary. Safe to
  /// call from the UI isolate while the worker is mid-generate (thread-safe
  /// atomic in native code).
  void cancel() {
    if (_closed) return;
    _cancelFn(Pointer<SipllmCtx>.fromAddress(_ctxAddress));
  }

  /// Compute an L2-normalized embedding for [text].
  Future<Float32List> embed(
    String text, {
    Pooling pooling = Pooling.last,
  }) async {
    if (_closed) throw SipllmException('runtime is closed');
    if (_busy) throw SipllmException('runtime is busy');
    _busy = true;
    final port = ReceivePort();
    final completer = Completer<Float32List>();
    void fail(String why) {
      if (!completer.isCompleted) completer.completeError(SipllmException(why));
      port.close();
    }
    _inflight.add(fail);
    port.listen((message) {
      _inflight.remove(fail);
      final m = message as Map;
      if (m['embedding'] != null) {
        completer.complete(m['embedding'] as Float32List);
      } else {
        completer.completeError(
            SipllmException((m['error'] ?? 'embed failed') as String));
      }
      port.close();
    });
    _toWorker.send({
      'cmd': 'embed',
      'text': text,
      'pooling': pooling.code,
      'port': port.sendPort,
    });
    try {
      return await completer.future;
    } finally {
      _busy = false;
    }
  }

  /// One request/reply exchange with the worker. Replies for which [isFinal]
  /// is false go to [onMessage] (streamed tokens). Fails if the worker dies or
  /// answers with an error.
  Future<Map<Object?, Object?>> _request(
    Map<String, Object?> msg, {
    void Function(Map<Object?, Object?> m)? onMessage,
    bool Function(Map<Object?, Object?> m)? isFinal,
  }) {
    if (_closed) return Future.error(SipllmException('runtime is closed'));
    final port = ReceivePort();
    final done = Completer<Map<Object?, Object?>>();
    void fail(String why) {
      if (!done.isCompleted) done.completeError(SipllmException(why));
      port.close();
    }

    _inflight.add(fail);
    port.listen((message) {
      final m = message as Map<Object?, Object?>;
      if (isFinal != null && !isFinal(m)) {
        onMessage?.call(m);
        return;
      }
      _inflight.remove(fail);
      port.close();
      if (m['error'] != null) {
        done.completeError(SipllmException(m['error'] as String));
      } else {
        done.complete(m);
      }
    });
    _toWorker.send({...msg, 'port': port.sendPort});
    return done.future;
  }

  /// Clear KV cache / conversation state.
  void reset() {
    if (_closed) return;
    _toWorker.send({'cmd': 'reset'});
  }

  /// Free the native context and shut down the worker isolate.
  ///
  /// An in-flight generation or agent step is cancelled first. The worker
  /// handles `close` only once the native call returns, and a long prompt's
  /// prefill cannot be interrupted, so this waits for the worker's reply
  /// rather than killing the isolate with the context (and its weights) still
  /// allocated. After [timeout] the isolate is killed anyway.
  Future<void> close({Duration timeout = const Duration(seconds: 120)}) async {
    if (_closed) return;
    if (_busy) cancel();
    _closed = true;
    if (_workerGone) return;
    final reply = ReceivePort();
    _toWorker.send({'cmd': 'close', 'port': reply.sendPort});
    try {
      await reply.first.timeout(timeout);
    } on TimeoutException {
      _isolate.kill(priority: Isolate.immediate);
    } finally {
      reply.close();
    }
  }
}
