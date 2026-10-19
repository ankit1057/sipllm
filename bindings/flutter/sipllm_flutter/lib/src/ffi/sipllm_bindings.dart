// ignore_for_file: library_private_types_in_public_api
// sipllm_bindings.dart — 1:1 Dart FFI mirror of ffi/sipllm_ffi.h.
//
// Struct field order and types MUST match the C header exactly; Dart FFI applies
// the platform's natural alignment, which is the same as the C compiler's, so
// no manual padding is needed. Hand-written (not ffigen) so the ABI is auditable
// against the header in one place.
import 'dart:ffi';

import 'package:ffi/ffi.dart';

// ---- opaque handle --------------------------------------------------------
final class SipllmCtx extends Opaque {}

// ---- sipllm_params --------------------------------------------------------
final class SipllmParamsC extends Struct {
  @Uint64()
  external int ramBudgetBytes;
  @Int32()
  external int threads;
  @Int32()
  external int maxCtx;
  @Int32()
  external int nBuffers;
  @Int32()
  external int useMmap;
  @Int32()
  external int asyncPrefetch;
  @Int32()
  external int fastQuant;
  @Int32()
  external int streamLmHead;
  @Int32()
  external int residencyFp32;
  @Int32()
  external int forceBudget;
  @Int32()
  external int schedulePolicy;
  @Int32()
  external int specDraft;
}

// ---- sipllm_sampler -------------------------------------------------------
final class SipllmSamplerC extends Struct {
  @Float()
  external double temperature;
  @Int32()
  external int topK;
  @Float()
  external double topP;
  @Float()
  external double repeatPenalty;
  @Int32()
  external int repeatLastN;
  @Uint64()
  external int seed;
}

// ---- sipllm_model_info ----------------------------------------------------
final class SipllmModelInfoC extends Struct {
  @Int32()
  external int nLayers;
  @Int32()
  external int nHeads;
  @Int32()
  external int nKvHeads;
  @Int64()
  external int dim;
  @Int64()
  external int vocabSize;
  @Int64()
  external int ctxLen;
  @Int32()
  external int tokenizerKind;
  @Array<Uint8>(32)
  external Array<Uint8> arch;
}

// ---- sipllm_stats ---------------------------------------------------------
final class SipllmStatsC extends Struct {
  @Double()
  external double loadS;
  @Double()
  external double ttftS;
  @Double()
  external double prefillS;
  @Double()
  external double decodeS;
  @Double()
  external double prefillTokS;
  @Double()
  external double decodeTokS;
  @Uint64()
  external int peakRssBytes;
  @Uint64()
  external int weightsResidentBytes;
  @Uint64()
  external int kvBytes;
  @Uint64()
  external int bytesRead;
  @Uint64()
  external int prefetchHits;
  @Uint64()
  external int prefetchMisses;
  @Int32()
  external int pinnedLayers;
  @Int32()
  external int nLayers;
  @Int32()
  external int promptTokens;
  @Int32()
  external int genTokens;
  @Int32()
  external int ctxUsed;
  @Int32()
  external int ctxMax;
}

// ---- token callback -------------------------------------------------------
typedef SipllmTokenCbNative = Int32 Function(
    Pointer<Utf8> piece, Int64 tokenId, Pointer<Void> user);

// ---- native function typedefs --------------------------------------------
typedef _ParamsDefaultNative = Void Function(Pointer<SipllmParamsC>);
typedef _ParamsDefaultDart = void Function(Pointer<SipllmParamsC>);

typedef _SamplerDefaultNative = Void Function(Pointer<SipllmSamplerC>);
typedef _SamplerDefaultDart = void Function(Pointer<SipllmSamplerC>);

typedef _OpenNative = Pointer<SipllmCtx> Function(
    Pointer<Utf8>, Pointer<SipllmParamsC>, Pointer<Utf8>, Int32);
typedef _OpenDart = Pointer<SipllmCtx> Function(
    Pointer<Utf8>, Pointer<SipllmParamsC>, Pointer<Utf8>, int);

typedef _CloseNative = Void Function(Pointer<SipllmCtx>);
typedef _CloseDart = void Function(Pointer<SipllmCtx>);

typedef _ModelInfoNative = Int32 Function(
    Pointer<SipllmCtx>, Pointer<SipllmModelInfoC>);
typedef _ModelInfoDart = int Function(
    Pointer<SipllmCtx>, Pointer<SipllmModelInfoC>);

typedef _GetThreadsNative = Int32 Function(Pointer<SipllmCtx>);
typedef _GetThreadsDart = int Function(Pointer<SipllmCtx>);

typedef _GenerateNative = Int32 Function(
    Pointer<SipllmCtx>,
    Pointer<Utf8>,
    Int32,
    Pointer<SipllmSamplerC>,
    Pointer<NativeFunction<SipllmTokenCbNative>>,
    Pointer<Void>,
    Pointer<SipllmStatsC>,
    Pointer<Utf8>,
    Int32);
typedef _GenerateDart = int Function(
    Pointer<SipllmCtx>,
    Pointer<Utf8>,
    int,
    Pointer<SipllmSamplerC>,
    Pointer<NativeFunction<SipllmTokenCbNative>>,
    Pointer<Void>,
    Pointer<SipllmStatsC>,
    Pointer<Utf8>,
    int);

typedef _CancelNative = Void Function(Pointer<SipllmCtx>);
typedef _CancelDart = void Function(Pointer<SipllmCtx>);

typedef _ResetNative = Void Function(Pointer<SipllmCtx>);
typedef _ResetDart = void Function(Pointer<SipllmCtx>);

typedef _EmbedDimNative = Int32 Function(Pointer<SipllmCtx>);
typedef _EmbedDimDart = int Function(Pointer<SipllmCtx>);

typedef _EmbedNative = Int32 Function(Pointer<SipllmCtx>, Pointer<Utf8>, Int32,
    Pointer<Float>, Int32, Pointer<Utf8>, Int32);
typedef _EmbedDart = int Function(Pointer<SipllmCtx>, Pointer<Utf8>, int,
    Pointer<Float>, int, Pointer<Utf8>, int);

typedef _VersionNative = Pointer<Utf8> Function();
typedef _VersionDart = Pointer<Utf8> Function();

typedef _IntVoidNative = Int32 Function();
typedef _IntVoidDart = int Function();

typedef _InfoNative = Pointer<Utf8> Function();
typedef _InfoDart = Pointer<Utf8> Function();

typedef _OptimalThreadsNative = Int32 Function(Pointer<Utf8>, Uint64);
typedef _OptimalThreadsDart = int Function(Pointer<Utf8>, int);

typedef _SetLogNative = Void Function(Int32);
typedef _SetLogDart = void Function(int);

/// Resolves and holds every native symbol for one loaded `libsipllm_ffi`.
class SipllmBindings {
  SipllmBindings(this._lib)
      : paramsDefault =
            _lib.lookupFunction<_ParamsDefaultNative, _ParamsDefaultDart>(
                'sipllm_params_default'),
        samplerDefault =
            _lib.lookupFunction<_SamplerDefaultNative, _SamplerDefaultDart>(
                'sipllm_sampler_default'),
        open = _lib.lookupFunction<_OpenNative, _OpenDart>('sipllm_open'),
        close = _lib.lookupFunction<_CloseNative, _CloseDart>('sipllm_close'),
        getModelInfo = _lib.lookupFunction<_ModelInfoNative, _ModelInfoDart>(
            'sipllm_get_model_info'),
        getThreads = _lib.lookupFunction<_GetThreadsNative, _GetThreadsDart>(
            'sipllm_get_threads'),
        generate =
            _lib.lookupFunction<_GenerateNative, _GenerateDart>('sipllm_generate'),
        cancel = _lib.lookupFunction<_CancelNative, _CancelDart>('sipllm_cancel'),
        reset = _lib.lookupFunction<_ResetNative, _ResetDart>('sipllm_reset'),
        embedDim =
            _lib.lookupFunction<_EmbedDimNative, _EmbedDimDart>('sipllm_embed_dim'),
        embed = _lib.lookupFunction<_EmbedNative, _EmbedDart>('sipllm_embed'),
        version =
            _lib.lookupFunction<_VersionNative, _VersionDart>('sipllm_version'),
        vulkanCompiled = _lib.lookupFunction<_IntVoidNative, _IntVoidDart>(
            'sipllm_vulkan_compiled'),
        vulkanAvailable = _lib.lookupFunction<_IntVoidNative, _IntVoidDart>(
            'sipllm_vulkan_available'),
        vulkanInfo =
            _lib.lookupFunction<_InfoNative, _InfoDart>('sipllm_vulkan_info'),
        hardwareConcurrency = _lib.lookupFunction<_IntVoidNative, _IntVoidDart>(
            'sipllm_hardware_concurrency'),
        optimalThreads =
            _lib.lookupFunction<_OptimalThreadsNative, _OptimalThreadsDart>(
                'sipllm_optimal_threads'),
        setLogLevel =
            _lib.lookupFunction<_SetLogNative, _SetLogDart>('sipllm_set_log_level');

  final DynamicLibrary _lib;
  DynamicLibrary get library => _lib;

  final _ParamsDefaultDart paramsDefault;
  final _SamplerDefaultDart samplerDefault;
  final _OpenDart open;
  final _CloseDart close;
  final _ModelInfoDart getModelInfo;
  final _GetThreadsDart getThreads;
  final _GenerateDart generate;
  final _CancelDart cancel;
  final _ResetDart reset;
  final _EmbedDimDart embedDim;
  final _EmbedDart embed;
  final _VersionDart version;
  final _IntVoidDart vulkanCompiled;
  final _IntVoidDart vulkanAvailable;
  final _InfoDart vulkanInfo;
  final _IntVoidDart hardwareConcurrency;
  final _OptimalThreadsDart optimalThreads;
  final _SetLogDart setLogLevel;
}

// Schedule policy + pooling enum constants (mirror sipllm_ffi.h).
const int kSchedStatic = 0;
const int kSchedFixed8 = 1;
const int kSchedFixed16 = 2;
const int kSchedFixed32 = 3;
const int kSchedProportional2 = 4;
const int kSchedProportional4 = 5;
const int kSchedAdaptive = 6;

const int kPoolLast = 0;
const int kPoolMean = 1;
