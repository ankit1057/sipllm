// sipllm_isolate.dart — the worker isolate that owns a native sipllm_ctx.
//
// Inference is synchronous in native code and would block the UI isolate, so it
// runs here. Tokens stream back over SendPorts; the token callback is an
// isolateLocal NativeCallable invoked synchronously by the engine on this
// isolate's thread while generate() is in flight. Cancellation does NOT go
// through this isolate (it is busy inside generate); the main isolate calls the
// thread-safe sipllm_cancel() directly on the shared ctx address.
import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../ffi/native_library.dart';
import '../ffi/sipllm_bindings.dart';
import 'sipllm_types.dart';
import 'utf8_pieces.dart';

// A C string from the engine as text. Malformed UTF-8 (a reply cut at
// max-tokens mid-character, an error buffer truncated mid-character) becomes
// U+FFFD instead of throwing: toDartString() is strict.
String _text(Pointer<Utf8> p) => p == nullptr
    ? ''
    : utf8.decode(p.cast<Uint8>().asTypedList(p.length), allowMalformed: true);

// The bytes of a token piece, copied out of native memory.
Uint8List _pieceBytes(Pointer<Utf8> p) =>
    p == nullptr ? Uint8List(0) : Uint8List.fromList(p.cast<Uint8>().asTypedList(p.length));

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

  // Agents created on this context, by id. Freed before the context.
  final agents = <int, Pointer<SipllmAgentC>>{};
  var nextAgent = 1;

  commands.listen((message) {
    final m = message as Map;
    switch (m['cmd'] as String) {
      case 'generate':
        _generate(b, ctx, m);
      case 'embed':
        _embed(b, ctx, m);
      case 'reset':
        b.reset(ctx);
      case 'agent_new':
        final port = m['port'] as SendPort;
        final r = _agentNew(b, ctx, m);
        if (r.agent == nullptr) {
          port.send({'error': r.error ?? 'sipllm_agent_new failed'});
        } else {
          final id = nextAgent++;
          agents[id] = r.agent;
          port.send({'agent': id});
        }
      case 'agent_start':
        _agentStart(b, agents[m['agent'] as int], m);
      case 'agent_step':
        _agentStep(b, agents[m['agent'] as int], m);
      case 'agent_submit':
        final port = m['port'] as SendPort;
        final a = agents[m['agent'] as int];
        if (a == null) {
          port.send({'error': 'agent was freed'});
        } else {
          _agentSubmit(b, a, m);
        }
      case 'agent_free':
        final a = agents.remove(m['agent'] as int);
        if (a != null) b.agentFree(a);
      case 'close':
        for (final a in agents.values) {
          b.agentFree(a);
        }
        agents.clear();
        b.close(ctx);
        commands.close();
        // The context is gone: tell the main isolate so it can stop waiting.
        (m['port'] as SendPort?)?.send({'closed': true});
        Isolate.exit();
    }
  });
}

class _AgentNewResult {
  const _AgentNewResult(this.agent, this.error);
  final Pointer<SipllmAgentC> agent;
  final String? error;
}

_AgentNewResult _agentNew(
    SipllmBindings b, Pointer<SipllmCtx> ctx, Map<Object?, Object?> m) {
  final arena = Arena();
  try {
    final p = arena<SipllmAgentParamsC>();
    b.agentParamsDefault(p);
    p.ref.maxSteps = m['maxSteps'] as int;
    p.ref.maxNewTokens = m['maxNewTokens'] as int;
    p.ref.toolFormat = m['toolFormat'] as int;
    p.ref.stopOnToolError = (m['stopOnToolError'] as bool) ? 1 : 0;
    final preamble = m['systemPreamble'] as String?;
    if (preamble != null) {
      p.ref.systemPreamble = preamble.toNativeUtf8(allocator: arena);
    }
    final errC = arena.allocate<Uint8>(512).cast<Utf8>();
    errC.cast<Uint8>().value = 0;
    final a = b.agentNew(ctx, p, errC, 512);
    if (a == nullptr) return _AgentNewResult(a, _text(errC));
    // Tools: [[name, description, [[param, type, required, description], ...]], ...]
    for (final t in (m['tools'] as List)) {
      final tool = t as List;
      final nameC = (tool[0] as String).toNativeUtf8(allocator: arena);
      final descC = (tool[1] as String).toNativeUtf8(allocator: arena);
      var ok = b.agentAddTool(a, nameC, descC) == 0;
      for (final pr in (tool[2] as List)) {
        final param = pr as List;
        ok = ok &&
            b.agentAddParam(
                    a,
                    nameC,
                    (param[0] as String).toNativeUtf8(allocator: arena),
                    param[1] as int,
                    (param[2] as bool) ? 1 : 0,
                    (param[3] as String).toNativeUtf8(allocator: arena)) ==
                0;
      }
      if (!ok) {
        b.agentFree(a);
        return _AgentNewResult(nullptr, 'invalid tool declaration: ${tool[0]}');
      }
    }
    return _AgentNewResult(a, null);
  } finally {
    arena.releaseAll();
  }
}

void _agentStart(
    SipllmBindings b, Pointer<SipllmAgentC>? a, Map<Object?, Object?> m) {
  final port = m['port'] as SendPort;
  if (a == null) {
    port.send({'error': 'agent was freed'});
    return;
  }
  final arena = Arena();
  try {
    final errC = arena.allocate<Uint8>(512).cast<Utf8>();
    errC.cast<Uint8>().value = 0;
    final rc = b.agentStart(
        a, (m['goal'] as String).toNativeUtf8(allocator: arena), errC, 512);
    port.send(rc == 0 ? {'started': true} : {'error': _text(errC)});
  } finally {
    arena.releaseAll();
  }
}

void _agentStep(
    SipllmBindings b, Pointer<SipllmAgentC>? a, Map<Object?, Object?> m) {
  final port = m['port'] as SendPort;
  if (a == null) {
    port.send({'error': 'agent was freed'});
    return;
  }
  final arena = Arena();
  // Pieces can split a character: decode across them (utf8_pieces.dart).
  final pieces = Utf8PieceDecoder();
  final callback = NativeCallable<SipllmTokenCbNative>.isolateLocal(
    (Pointer<Utf8> piece, int tokenId, Pointer<Void> user) {
      port.send({'token': pieces.add(_pieceBytes(piece)), 'id': tokenId});
      return 1;
    },
    exceptionalReturn: 0,
  );
  try {
    final statsC = arena<SipllmStatsC>();
    final errC = arena.allocate<Uint8>(512).cast<Utf8>();
    errC.cast<Uint8>().value = 0;
    // NULL sampler: greedy. Tool names and arguments must not be sampled.
    final ev = b.agentStep(
        a, nullptr, callback.nativeFunction, nullptr, statsC, errC, 512);
    final reply = <String, Object?>{
      'event': ev,
      'steps': b.agentSteps(a),
      'text': _text(b.agentText(a)),
      'stats': _readStats(statsC.ref).toMap(),
    };
    if (ev == kAgentError) reply['error'] = _text(errC);
    if (ev == kAgentToolCall) {
      reply['name'] = _text(b.agentToolName(a));
      reply['raw'] = _text(b.agentToolArgsJson(a));
      final n = b.agentArgCount(a);
      reply['args'] = [
        for (var i = 0; i < n; i++)
          [_text(b.agentArgKey(a, i)), _text(b.agentArgValue(a, i))]
      ];
    }
    port.send({'step': reply});
  } catch (e) {
    port.send({'error': '$e'});
  } finally {
    callback.close();
    arena.releaseAll();
  }
}

void _agentSubmit(
    SipllmBindings b, Pointer<SipllmAgentC> a, Map<Object?, Object?> m) {
  final port = m['port'] as SendPort;
  final arena = Arena();
  try {
    final rc = b.agentSubmitResult(
        a,
        (m['result'] as String).toNativeUtf8(allocator: arena),
        (m['ok'] as bool) ? 1 : 0);
    port.send({'submitted': rc, 'stop': b.agentStopReason(a), 'steps': b.agentSteps(a)});
  } finally {
    arena.releaseAll();
  }
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
      return _OpenResult(ctx, _text(errC));
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
  // us via sipllm_cancel(). Streaming a piece is just a port.send. Every token
  // is sent (so token counts stay right); a piece that only starts a
  // multi-byte character sends '' and the piece completing it sends it whole.
  final pieces = Utf8PieceDecoder();
  final callback = NativeCallable<SipllmTokenCbNative>.isolateLocal(
    (Pointer<Utf8> piece, int tokenId, Pointer<Void> user) {
      port.send({'token': pieces.add(_pieceBytes(piece)), 'id': tokenId});
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
      port.send({'error': _text(errC)});
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
      port.send({'error': _text(errC)});
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
