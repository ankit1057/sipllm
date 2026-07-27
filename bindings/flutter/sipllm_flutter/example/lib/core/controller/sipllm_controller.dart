import 'dart:async';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sipllm_flutter/sipllm_flutter.dart';

import '../chat/prompt_format.dart';
import '../models/benchmark.dart';
import '../models/chat_message.dart';
import '../models/managed_model.dart';
import '../models/model_catalog.dart';

/// The single source of truth for the example app: device capabilities, the
/// model manager (download / import / delete), the loaded inference runtime,
/// streaming chat, an optional embedding-backed memory, phone->watch transfer,
/// and benchmark capture. UI widgets are thin `Consumer`s over this.
class SipllmController extends ChangeNotifier {
  // ---- device / engine ----------------------------------------------------
  DeviceProfile? deviceProfile;
  SipllmDevice? _device;
  String engineVersion = '';
  int hardwareConcurrency = 0;
  bool vulkanCompiled = false;
  bool vulkanAvailable = false;
  String vulkanInfo = '';

  // ---- model manager -------------------------------------------------------
  String? _modelsDir;
  final Map<String, ManagedModel> _models = {};
  final DownloadManager _downloads = DownloadManager(maxConnections: 6);

  List<ManagedModel> get models {
    final list = _models.values.toList();
    list.sort((a, b) {
      if (a.isInstalled != b.isInstalled) return a.isInstalled ? -1 : 1;
      return a.name.compareTo(b.name);
    });
    return list;
  }

  int get installedCount =>
      _models.values.where((m) => m.isInstalled).length;
  int get totalStorageBytes => _models.values
      .where((m) => m.isInstalled)
      .fold(0, (sum, m) => sum + m.sizeBytes);

  // ---- runtime -------------------------------------------------------------
  SipllmRuntime? _rt;
  SipllmRuntime? get runtime => _rt;
  ManagedModel? loadedModel;
  bool get isModelLoaded => _rt != null;
  bool loading = false;

  // ---- generation config ---------------------------------------------------
  int ramBudgetMiB = 0; // 0 = unlimited streaming
  int threads = 0; // 0 = hardware_concurrency
  bool fastQuant = true;
  bool streamLmHead = false;
  SchedulePolicy schedulePolicy = SchedulePolicy.proportional2;
  double temperature = 0.7;
  double topP = 0.95;
  int maxTokens = 256;
  int maxCtx = 2048;
  String systemPrompt = 'You are SipLLM, a concise, helpful on-device assistant.';

  // ---- chat ----------------------------------------------------------------
  final List<ChatMessage> messages = [];
  bool generating = false;
  StreamSubscription<SipllmToken>? _genSub;
  SipllmStats? get lastStats => _rt?.lastStats;

  // ---- RAG memory ----------------------------------------------------------
  bool useRag = false;
  EmbeddingStore? _memory;

  // ---- wear ----------------------------------------------------------------
  final WearTransfer wear = WearTransfer();

  // ---- game overlay state --------------------------------------------------
  String? activeGame;
  bool gameMinimized = false;

  void launchGame(String gameName) {
    activeGame = gameName;
    gameMinimized = false;
    notifyListeners();
  }

  void minimizeGame() {
    gameMinimized = true;
    notifyListeners();
  }

  void resumeGame() {
    gameMinimized = false;
    notifyListeners();
  }

  void closeGame() {
    activeGame = null;
    gameMinimized = false;
    notifyListeners();
  }

  // ---- benchmarks ----------------------------------------------------------
  final List<BenchmarkResult> benchmarks = [];

  bool _initialized = false;
  bool get initialized => _initialized;

  Future<String> _resolveDownloadsSipLlmDir() async {
    Directory? base;
    if (Platform.isAndroid) {
      try {
        base = await getExternalStorageDirectory();
      } catch (_) {}
    } else {
      try {
        base = await getDownloadsDirectory();
      } catch (_) {}
    }
    base ??= await getApplicationSupportDirectory();
    final target = Directory(p.join(base.path, 'SipLLM'));
    if (!target.existsSync()) {
      await target.create(recursive: true);
    }
    return target.path;
  }

  Future<void> init() async {
    if (_initialized) return;
    try {
      _device = SipllmDevice.load();
      _device!.setLogLevel(1);
      engineVersion = _device!.engineVersion;
      hardwareConcurrency = _device!.hardwareConcurrency;
      vulkanCompiled = _device!.vulkanCompiled;
      vulkanAvailable = _device!.vulkanAvailable;
      vulkanInfo = _device!.vulkanInfo;
    } catch (e) {
      vulkanInfo = 'engine load failed: $e';
    }
    try {
      deviceProfile = await ArchDetector().detect();
      // A watch defaults to a tight streaming budget.
      if (deviceProfile!.isWearOs && ramBudgetMiB == 0) {
        ramBudgetMiB = deviceProfile!.suggestedRamBudgetBytes ~/ (1024 * 1024);
      }
    } catch (_) {}

    _modelsDir = await _resolveDownloadsSipLlmDir();
    await Directory(_modelsDir!).create(recursive: true);
    await _rebuildModelIndex();

    _initialized = true;
    notifyListeners();
  }

  bool get isWearOs => deviceProfile?.isWearOs ?? false;

  // ---- model index ---------------------------------------------------------
  Future<void> _rebuildModelIndex() async {
    final support = await getApplicationSupportDirectory();
    final legacyDir = p.join(support.path, 'models');

    // Start from the catalog...
    for (final c in kModelCatalog) {
      final path = p.join(_modelsDir!, c.localFileName);
      final legacyPath = p.join(legacyDir, c.localFileName);

      final existing = _models[c.id];
      if (existing != null && existing.isBusy) continue; // don't clobber a download

      final file = File(path);
      final legacyFile = File(legacyPath);

      bool installed = false;
      String targetPath = path;
      int size = c.approxBytes;

      if (file.existsSync() && file.lengthSync() > 0) {
        installed = true;
        targetPath = path;
        size = file.lengthSync();
      } else if (legacyFile.existsSync() && legacyFile.lengthSync() > 0) {
        installed = true;
        targetPath = legacyPath;
        size = legacyFile.lengthSync();
      }

      _models[c.id] = ManagedModel(
        id: c.id,
        name: c.name,
        localPath: targetPath,
        catalog: c,
        status: installed ? ModelStatus.installed : ModelStatus.available,
        sizeBytes: size,
      );
    }

    // Index all .gguf files in Downloads/SipLLM and legacy models dir
    final scanDirs = [_modelsDir!, legacyDir];
    for (final dPath in scanDirs) {
      final dir = Directory(dPath);
      if (dir.existsSync()) {
        for (final e in dir.listSync()) {
          if (e is! File || !e.path.endsWith('.gguf')) continue;
          final name = p.basenameWithoutExtension(e.path);
          if (_models.values.any((m) => m.localPath == e.path)) continue;
          _models[name] = ManagedModel(
            id: name,
            name: name,
            localPath: e.path,
            status: ModelStatus.installed,
            sizeBytes: e.lengthSync(),
            imported: true,
          );
        }
      }
    }
  }

  // ---- download ------------------------------------------------------------
  Future<void> download(ManagedModel model) async {
    final c = model.catalog;
    if (c == null) return;
    model
      ..status = ModelStatus.downloading
      ..error = null;
    notifyListeners();

    final task = _downloads.enqueue(
      url: c.url,
      destPath: model.localPath,
      connections: 6,
    );
    model.task = task;
    task.progress.listen((prog) {
      model.progress = prog;
      if (model.status == ModelStatus.downloading) notifyListeners();
    });
    try {
      await task.done;
      model
        ..status = ModelStatus.installed
        ..sizeBytes = File(model.localPath).existsSync()
            ? File(model.localPath).lengthSync()
            : c.approxBytes
        ..task = null;
    } catch (e) {
      model
        ..status = ModelStatus.failed
        ..error = '$e'
        ..task = null;
    }
    notifyListeners();
  }

  void pauseDownload(ManagedModel model) {
    model.task?.pause();
    model.status = ModelStatus.paused;
    notifyListeners();
  }

  Future<void> resumeDownload(ManagedModel model) async {
    final c = model.catalog;
    if (c == null) return;

    if (model.task == null ||
        model.task!.state == DownloadState.failed ||
        model.task!.state == DownloadState.paused) {
      await download(model);
    } else {
      model.status = ModelStatus.downloading;
      notifyListeners();
      model.task!.resume();
    }
  }

  void cancelDownload(ManagedModel model) {
    model.task?.cancel();
    model
      ..task = null
      ..progress = null
      ..status = ModelStatus.available;
    notifyListeners();
  }

  // ---- import / delete -----------------------------------------------------
  Future<String?> importModel() async {
    final res = await FilePicker.platform.pickFiles(
      type: FileType.any,
      withData: false,
    );
    final picked = res?.files.single.path;
    if (picked == null) return null;
    if (!picked.toLowerCase().endsWith('.gguf')) {
      return 'Not a .gguf file';
    }
    final name = p.basename(picked);
    final dest = p.join(_modelsDir!, name);
    final id = p.basenameWithoutExtension(name);
    _models[id] = ManagedModel(
      id: id,
      name: id,
      localPath: dest,
      status: ModelStatus.importing,
      imported: true,
    );
    notifyListeners();
    try {
      await File(picked).copy(dest);
      _models[id]!
        ..status = ModelStatus.installed
        ..sizeBytes = File(dest).lengthSync();
    } catch (e) {
      _models[id]!
        ..status = ModelStatus.failed
        ..error = '$e';
    }
    notifyListeners();
    return null;
  }

  Future<void> deleteModel(ManagedModel model) async {
    if (loadedModel?.id == model.id) await unloadModel();
    final f = File(model.localPath);
    if (f.existsSync()) await f.delete();
    if (model.imported) {
      _models.remove(model.id);
    } else {
      model
        ..status = ModelStatus.available
        ..sizeBytes = model.catalog?.approxBytes ?? 0
        ..progress = null;
    }
    notifyListeners();
  }

  /// Lightweight integrity check: the file exists, is non-empty, and starts
  /// with the GGUF magic.
  Future<bool> verifyModel(ManagedModel model) async {
    final f = File(model.localPath);
    if (!f.existsSync() || f.lengthSync() < 4) return false;
    final head = await f.openRead(0, 4).first;
    return head.length >= 4 &&
        head[0] == 0x47 && head[1] == 0x47 && head[2] == 0x55 && head[3] == 0x46; // "GGUF"
  }

  // ---- runtime lifecycle ---------------------------------------------------
  SipllmParams get params => SipllmParams(
        ramBudgetBytes: ramBudgetMiB * 1024 * 1024,
        threads: threads,
        maxCtx: maxCtx,
        fastQuant: fastQuant,
        streamLmHead: streamLmHead,
        schedulePolicy: schedulePolicy,
      );

  Future<void> loadModel(ManagedModel model) async {
    if (!model.isInstalled) return;
    await unloadModel();
    loading = true;
    notifyListeners();
    try {
      _rt = await SipllmRuntime.open(model.localPath, params: params);
      loadedModel = model;
      _template = PromptTemplate.forModel(model.id);
      _openMemory(_rt!.modelInfo.dim);
    } catch (e) {
      _rt = null;
      loadedModel = null;
      rethrow;
    } finally {
      loading = false;
      notifyListeners();
    }
  }

  Future<void> unloadModel() async {
    await _genSub?.cancel();
    _genSub = null;
    generating = false;
    final rt = _rt;
    _rt = null;
    loadedModel = null;
    if (rt != null) await rt.close();
    notifyListeners();
  }

  // ---- chat ----------------------------------------------------------------
  PromptTemplate _template = PromptTemplate.chatml;

  Future<void> send(String text) async {
    final rt = _rt;
    if (rt == null || generating || text.trim().isEmpty) return;

    messages.add(ChatMessage(text: text.trim(), isUser: true));
    final assistant = ChatMessage(text: '', isUser: false, streaming: true);
    generating = true;
    notifyListeners();

    var system = systemPrompt;
    if (useRag && _memory != null && _memory!.length > 0) {
      try {
        final q = await rt.embed(text);
        final hits = _memory!.search(q, topK: 3);
        if (hits.isNotEmpty) {
          system +=
              '\n\nRelevant memory:\n${hits.map((h) => '- ${h.text}').join('\n')}';
        }
      } catch (_) {}
    }

    final prompt = _template.build(system, List.of(messages));
    messages.add(assistant);
    notifyListeners();

    rt.reset();
    final sampler = SipllmSampler(
      temperature: temperature,
      topP: topP,
      topK: 40,
      repeatPenalty: 1.1,
    );

    final completer = Completer<void>();
    _genSub = rt.generate(prompt, maxNew: maxTokens, sampler: sampler).listen(
      (tok) {
        assistant.text += tok.piece;
        notifyListeners();
      },
      onDone: () async {
        assistant
          ..streaming = false
          ..stats = rt.lastStats;
        generating = false;
        _genSub = null;
        if (useRag && _memory != null && assistant.text.trim().isNotEmpty) {
          try {
            final e = await rt.embed(assistant.text);
            _memory!.add(text: assistant.text.trim(), embedding: e);
          } catch (_) {}
        }
        notifyListeners();
        if (!completer.isCompleted) completer.complete();
      },
      onError: (Object e) {
        assistant
          ..text += '\n\n[error: $e]'
          ..streaming = false;
        generating = false;
        _genSub = null;
        notifyListeners();
        if (!completer.isCompleted) completer.complete();
      },
    );
    return completer.future;
  }

  void cancelGeneration() => _rt?.cancel();

  Stream<SipllmToken> generateStream(String prompt, {int? maxNewTokens}) {
    final rt = _rt;
    if (rt == null) return const Stream.empty();
    rt.reset();
    final sampler = SipllmSampler(
      temperature: temperature,
      topP: topP,
      topK: 40,
      repeatPenalty: 1.1,
    );
    generating = true;
    notifyListeners();
    return rt.generate(prompt, maxNew: maxNewTokens ?? maxTokens, sampler: sampler);
  }

  void resetChat() {
    _rt?.reset();
    messages.clear();
    notifyListeners();
  }

  // ---- RAG memory ----------------------------------------------------------
  void _openMemory(int dim) {
    _memory?.close();
    try {
      _memory = EmbeddingStore.open(
        p.join(_modelsDir!, 'memory-$dim.sqlite'),
        dim: dim,
      );
    } catch (_) {
      _memory = null;
    }
  }

  int get memorySize => _memory?.length ?? 0;
  void clearMemory() {
    _memory?.clear();
    notifyListeners();
  }

  // ---- benchmark -----------------------------------------------------------
  bool benchmarking = false;

  Future<BenchmarkResult?> runBenchmark({
    String prompt = 'Explain, in three sentences, why the sky is blue.',
    int tokens = 64,
  }) async {
    final rt = _rt;
    if (rt == null || generating || benchmarking) return null;
    benchmarking = true;
    notifyListeners();
    try {
      rt.reset();
      await rt
          .generate(prompt, maxNew: tokens, sampler: const SipllmSampler.greedy())
          .drain<void>();
      final stats = rt.lastStats;
      if (stats == null) return null;
      final result = BenchmarkResult(
        timestamp: DateTime.now(),
        modelName: loadedModel?.name ?? 'unknown',
        quant: loadedModel?.quant ?? '',
        deviceModel: deviceProfile?.model ?? 'unknown',
        arch: deviceProfile?.primaryArch.androidAbi ?? 'unknown',
        cores: deviceProfile?.cores ?? hardwareConcurrency,
        threads: rt.threads,
        schedulePolicy: schedulePolicy.name,
        ramBudgetBytes: ramBudgetMiB * 1024 * 1024,
        vulkan: vulkanAvailable ? 'active' : (vulkanCompiled ? 'compiled' : 'off'),
        engineVersion: engineVersion,
        stats: stats,
      );
      benchmarks.insert(0, result);
      return result;
    } finally {
      benchmarking = false;
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _genSub?.cancel();
    _rt?.close();
    _memory?.close();
    _downloads.disposeAll();
    super.dispose();
  }
}
