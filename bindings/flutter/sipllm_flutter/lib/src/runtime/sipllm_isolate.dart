// sipllm_isolate.dart — the worker isolate that owns a native sipllm_ctx.
//
// Inference is synchronous in native code and would block the UI isolate, so it
// runs here. Tokens stream back over SendPorts; the token callback is an
// isolateLocal NativeCallable invoked synchronously by the engine on this
// isolate's thread while generate() is in flight. Cancellation does NOT go
// through this isolate (it is busy inside generate); the main isolate calls the
// thread-safe sipllm_cancel() directly on the shared ctx address.
import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../ffi/native_library.dart';
import '../ffi/sipllm_bindings.dart';
import 'sipllm_types.dart';

/// Message sent to bootstrap the worker.
class SipllmWorkerInit {
  const SipllmWorkerInit(
      this.reply, this.libraryPath, this.modelPath, this.params, this.logLevel);
  final SendPort reply;
  final String? libraryPath;
  final String modelPath;
  final Map<String, Object> params;
  final int logLevel;
}

/// Entry point for [Isolate.spawn].
void sipllmWorkerMain(SipllmWorkerInit init) {
  late final SipllmBindings b;
  try {
    b = loadSipllmBindings(libraryPath: init.libraryPath);
  } catch (e) {
    init.reply.send({'error': 'failed to load libsipllm_ffi: $e'});
    return;
  }
  b.setLogLevel(init.logLevel);

  final open = _open(b, init.modelPath, SipllmParams.fromMap(init.params));
  if (open.ctx == nullptr) {
    init.reply.send({'error': open.error ?? 'sipllm_open failed'});
    return;
  }
  final ctx = open.ctx;

  final info = _readModelInfo(b, ctx);
  final threads = b.getThreads(ctx);

  final commands = ReceivePort();
  init.reply.send({
    'toWorker': commands.sendPort,
    'ctxAddress': ctx.address,
    'modelInfo': info.toMap(),
    'threads': threads,
  });

  commands.listen((message) {
    final m = message as Map;
    switch (m['cmd'] as String) {
      case 'generate':
        _generate(b, ctx, m);
      case 'embed':
        _embed(b, ctx, m);
      case 'reset':
        b.reset(ctx);
      case 'close':
        b.close(ctx);
        commands.close();
        Isolate.exit();
    }
  });
}

class _OpenResult {
  const _OpenResult(this.ctx, this.error);
  final Pointer<SipllmCtx> ctx;
  final String? error;
}

_OpenResult _open(SipllmBindings b, String modelPath, SipllmParams p) {
  final arena = Arena();
  try {
    final pc = arena<SipllmParamsC>();
    b.paramsDefault(pc);
    final r = pc.ref;
    r.ramBudgetBytes = p.ramBudgetBytes;
    r.threads = p.threads;
    r.maxCtx = p.maxCtx;
    r.nBuffers = p.nBuffers;
    r.useMmap = p.useMmap ? 1 : 0;
    r.asyncPrefetch = p.asyncPrefetch ? 1 : 0;
    r.fastQuant = p.fastQuant ? 1 : 0;
    r.streamLmHead = p.streamLmHead ? 1 : 0;
    r.residencyFp32 = p.residencyFp32 ? 1 : 0;
    r.forceBudget = p.forceBudget ? 1 : 0;
    r.schedulePolicy = p.schedulePolicy.code;
    r.specDraft = p.specDraft;

    final pathC = modelPath.toNativeUtf8(allocator: arena);
    final errC = arena.allocate<Uint8>(512).cast<Utf8>();
    errC.cast<Uint8>().value = 0;
    final ctx = b.open(pathC, pc, errC, 512);
    if (ctx == nullptr) {
      return _OpenResult(ctx, errC.toDartString());
    }
    return _OpenResult(ctx, null);
  } finally {
    arena.releaseAll();
  }
}

SipllmModelInfo _readModelInfo(SipllmBindings b, Pointer<SipllmCtx> ctx) {
  final arena = Arena();
  try {
    final mi = arena<SipllmModelInfoC>();
    b.getModelInfo(ctx, mi);
    final r = mi.ref;
    // arch is a fixed char[32]; read up to the NUL.
    final bytes = <int>[];
    for (var i = 0; i < 32; i++) {
      final c = r.arch[i];
      if (c == 0) break;
      bytes.add(c);
    }
    return SipllmModelInfo(
      arch: String.fromCharCodes(bytes),
      nLayers: r.nLayers,
      nHeads: r.nHeads,
      nKvHeads: r.nKvHeads,
      dim: r.dim,
      vocabSize: r.vocabSize,
      ctxLen: r.ctxLen,
      tokenizerKind: TokenizerKind.values[r.tokenizerKind.clamp(0, 2)],
    );
  } finally {
    arena.releaseAll();
  }
}

void _generate(SipllmBindings b, Pointer<SipllmCtx> ctx, Map<Object?, Object?> m) {
  final port = m['port'] as SendPort;
  final prompt = m['prompt'] as String;
  final maxNew = m['maxNew'] as int;
  final sampler = SipllmSampler.fromMap(m['sampler'] as Map<String, Object?>);
  final arena = Arena();

  // isolateLocal: invoked synchronously on this thread by the engine during
  // the blocking generate() call. Returning 1 continues; the main isolate stops
  // us via sipllm_cancel(). Streaming a piece is just a port.send.
  final callback = NativeCallable<SipllmTokenCbNative>.isolateLocal(
    (Pointer<Utf8> piece, int tokenId, Pointer<Void> user) {
      port.send({'token': piece.toDartString(), 'id': tokenId});
      return 1;
    },
    exceptionalReturn: 0,
  );

  try {
    final promptC = prompt.toNativeUtf8(allocator: arena);
    final samplerC = arena<SipllmSamplerC>();
    final s = samplerC.ref;
    s.temperature = sampler.temperature;
    s.topK = sampler.topK;
    s.topP = sampler.topP;
    s.repeatPenalty = sampler.repeatPenalty;
    s.repeatLastN = sampler.repeatLastN;
    s.seed = sampler.seed;
    final statsC = arena<SipllmStatsC>();
    final errC = arena.allocate<Uint8>(512).cast<Utf8>();
    errC.cast<Uint8>().value = 0;

    final n = b.generate(ctx, promptC, maxNew, samplerC, callback.nativeFunction,
        nullptr, statsC, errC, 512);
    if (n < 0) {
      port.send({'error': errC.toDartString()});
    } else {
      port.send({'done': true, 'stats': _readStats(statsC.ref).toMap()});
    }
  } catch (e) {
    port.send({'error': '$e'});
  } finally {
    callback.close();
    arena.releaseAll();
  }
}

SipllmStats _readStats(SipllmStatsC r) => SipllmStats(
      ttftSeconds: r.ttftS,
      prefillTokensPerSecond: r.prefillTokS,
      decodeTokensPerSecond: r.decodeTokS,
      peakRssBytes: r.peakRssBytes,
      weightsResidentBytes: r.weightsResidentBytes,
      kvBytes: r.kvBytes,
      bytesRead: r.bytesRead,
      prefetchHits: r.prefetchHits,
      prefetchMisses: r.prefetchMisses,
      pinnedLayers: r.pinnedLayers,
      nLayers: r.nLayers,
      promptTokens: r.promptTokens,
      genTokens: r.genTokens,
      ctxUsed: r.ctxUsed,
      ctxMax: r.ctxMax,
    );

void _embed(SipllmBindings b, Pointer<SipllmCtx> ctx, Map<Object?, Object?> m) {
  final port = m['port'] as SendPort;
  final text = m['text'] as String;
  final pooling = m['pooling'] as int;
  final arena = Arena();
  try {
    final dim = b.embedDim(ctx);
    if (dim <= 0) {
      port.send({'error': 'model has no embedding dimension'});
      return;
    }
    final textC = text.toNativeUtf8(allocator: arena);
    final out = arena.allocate<Float>(dim * sizeOf<Float>());
    final errC = arena.allocate<Uint8>(512).cast<Utf8>();
    errC.cast<Uint8>().value = 0;
    final rc = b.embed(ctx, textC, pooling, out, dim, errC, 512);
    if (rc < 0) {
      port.send({'error': errC.toDartString()});
      return;
    }
    final vec = Float32List(dim);
    for (var i = 0; i < dim; i++) {
      vec[i] = out[i];
    }
    port.send({'embedding': vec});
  } catch (e) {
    port.send({'error': '$e'});
  } finally {
    arena.releaseAll();
  }
}
