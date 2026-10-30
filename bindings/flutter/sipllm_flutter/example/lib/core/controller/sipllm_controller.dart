import 'dart:async';
import 'dart:convert';
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

  /// True while a model is being closed. Closing waits for the worker, which
  /// can take seconds when a prefill is in flight.
  bool unloading = false;
  Future<void>? _closing;

  // ---- generation config ---------------------------------------------------
  int ramBudgetMiB = 0; // 0 = automatic (half of available RAM, issue #78)
  int threads = 0; // 0 = automatic: [fastCores] if known, else hardware_concurrency

  /// The CPU's fast ("big") cores on Android, from cpufreq; 0 when every core
  /// has the same top speed or it could not be read. Used when [threads] is 0.
  int fastCores = 0;

  /// The thread count a model load asks for: the user's choice, else the fast
  /// cores, else 0 (the engine then uses every core).
  int get effectiveThreads => threads != 0 ? threads : fastCores;

  /// How the automatic thread count is shown next to the setting.
  String get autoThreadsLabel => fastCores > 0 ? 'auto: $fastCores fast cores' : 'auto';
  bool fastQuant = true;
  bool streamLmHead = false;
  SchedulePolicy schedulePolicy = SchedulePolicy.fixed16;
  double temperature = 0.7;
  double topP = 0.95;
  int maxTokens = 256;
  // 4096: a Phone-mode agent run holds the tools, the goal and one compact
  // screen per step (~170 tokens each) in one conversation.
  int maxCtx = 4096;
  String systemPrompt = 'You are SipLLM, a concise, helpful on-device assistant.';

  // ---- chat ----------------------------------------------------------------
  final List<ChatMessage> messages = [];
  bool generating = false;
  StreamSubscription<SipllmToken>? _genSub;
  // Bumped by every generation start and by unloadModel, so a generation's
  // late callbacks never clear the busy flag of a newer one.
  int _genEpoch = 0;
  // The chat turn being streamed, so unloadModel can close it.
  ChatMessage? _inflightReply;
  Completer<void>? _inflightDone;

  /// An agent run (Agent screen) is using the runtime. Chat, Playground,
  /// Arena and Benchmark cannot start meanwhile, and loading or unloading a
  /// model stops the run.
  bool agentRunning = false;

  void setAgentRunning(bool running) {
    if (agentRunning == running) return;
    agentRunning = running;
    notifyListeners();
  }

  /// Something is using the loaded model: a chat reply, a Playground or
  /// Arena completion, a benchmark, or an agent run.
  bool get runtimeInUse => generating || benchmarking || agentRunning;
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
    if (Platform.isAndroid) {
      try {
        fastCores = fastCoreCount(readCpuMaxFreqs());
      } catch (_) {
        fastCores = 0;
      }
    }

    _modelsDir = await _resolveDownloadsSipLlmDir();
    await Directory(_modelsDir!).create(recursive: true);
    try {
      final support = await getApplicationSupportDirectory();
      _legacyDir = p.join(support.path, 'models');
    } catch (_) {}
    _indexModels();
    await _loadHistory();

    _initialized = true;
    notifyListeners();
  }

  bool get isWearOs => deviceProfile?.isWearOs ?? false;

  // ---- model index ---------------------------------------------------------
  String? _legacyDir; // <app support>/models, where older builds kept models

  /// Points the index at [modelsDir] (and [legacyDir]) without init(); tests.
  @visibleForTesting
  void debugUseModelDirs(String modelsDir, {String? legacyDir}) {
    _modelsDir = modelsDir;
    _legacyDir = legacyDir;
    _initialized = true;
  }

  /// Re-read the models folders, so a GGUF imported or copied in after launch
  /// shows up and an imported one deleted outside the app goes away. Entries
  /// that are downloading (or paused), importing, failed, or loaded are left
  /// exactly as they are.
  Future<void> rescanModels() async {
    if (!_initialized || _modelsDir == null) return;
    try {
      _indexModels();
    } catch (_) {
      // An unreadable folder leaves the list as it was.
    }
    notifyListeners();
  }

  bool _keepAsIs(ManagedModel? m) =>
      m != null &&
      (m.status == ModelStatus.downloading ||
          m.status == ModelStatus.paused ||
          m.status == ModelStatus.importing ||
          m.task != null ||
          loadedModel?.id == m.id);

  // A second copy of a catalog model (the other of its two file names) is
  // listed, so its storage stays visible, but marked: never deleted here.
  String? _catalogNameFor(String path) {
    final base = p.basename(path);
    for (final c in kModelCatalog) {
      if (base != c.localFileName && base != p.basename(c.file)) continue;
      final m = _models[c.id];
      if (m != null && m.isInstalled && m.localPath != path) return m.name;
    }
    return null;
  }

  void _indexModels() {
    final legacyDir = _legacyDir;

    // Start from the catalog...
    for (final c in kModelCatalog) {
      final existing = _models[c.id];
      if (_keepAsIs(existing)) continue; // don't clobber a download or the loaded model

      // The catalog's file under the name a download gives it (<id>.gguf) or
      // under its Hugging Face name, as a browser or `adb push` saves it.
      final path = p.join(_modelsDir!, c.localFileName);
      bool installed = false;
      String targetPath = path;
      int size = c.approxBytes;
      for (final candidate in catalogFilePaths(c, _modelsDir!, legacyDir: legacyDir)) {
        final f = File(candidate);
        final len = f.existsSync() ? f.lengthSync() : 0;
        if (len <= 0) continue;
        // Listed under its own entry and loaded (or importing): leave it there.
        if (_models.values.any((m) => m.catalog == null && m.localPath == candidate && _keepAsIs(m))) continue;
        installed = true;
        targetPath = candidate;
        size = len;
        break;
      }

      // A failed download keeps its error until its file shows up.
      if (existing != null && existing.status == ModelStatus.failed && !installed) continue;
      // The file was listed as a separate model (e.g. a second copy whose
      // first copy is now gone): it is this entry's file, list it once.
      if (installed) {
        _models.removeWhere((_, m) => m.catalog == null && m.localPath == targetPath && !_keepAsIs(m));
      }
      final status = installed ? ModelStatus.installed : ModelStatus.available;
      if (existing != null &&
          existing.localPath == targetPath &&
          existing.status == status &&
          existing.sizeBytes == size) {
        continue; // unchanged: keep the same object
      }

      _models[c.id] = ManagedModel(
        id: c.id,
        name: c.name,
        localPath: targetPath,
        catalog: c,
        status: status,
        sizeBytes: size,
      );
    }

    // Index all .gguf files in Downloads/SipLLM and legacy models dir
    final scanDirs = [_modelsDir!, if (legacyDir != null) legacyDir];
    for (final dPath in scanDirs) {
      final dir = Directory(dPath);
      if (!dir.existsSync()) continue;
      for (final e in dir.listSync()) {
        if (e is! File || !e.path.endsWith('.gguf')) continue;
        if (_models.values.any((m) => m.localPath == e.path)) continue;
        final name = p.basenameWithoutExtension(e.path);
        // The same name in both folders: the entry already listed stays, so
        // repeated rescans do not flip between the two files.
        final existing = _models[name];
        if (existing != null && (_keepAsIs(existing) || File(existing.localPath).existsSync())) continue;
        final int size;
        try {
          size = e.lengthSync();
        } catch (_) {
          continue;
        }
        _models[name] = ManagedModel(
          id: name,
          name: name,
          localPath: e.path,
          status: ModelStatus.installed,
          sizeBytes: size,
          imported: true,
          duplicateOf: _catalogNameFor(e.path),
        );
      }
    }

    // An imported file deleted outside the app is no longer listed.
    _models.removeWhere((_, m) =>
        m.imported &&
        m.status == ModelStatus.installed &&
        !_keepAsIs(m) &&
        !File(m.localPath).existsSync());
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
  /// Copy a picked GGUF into the models folder. Never overwrites: a file of
  /// that name may be the loaded (memory-mapped) model or another entry's. The
  /// copy is written under a temporary name and renamed when complete, so a
  /// rescan never lists a half-copied file. Returns a message for the user,
  /// or null when there is nothing to say (cancelled, or imported).
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
    if (p.equals(picked, dest) || File(dest).existsSync()) {
      await rescanModels();
      return '"$name" is already in the models folder';
    }
    final key = 'importing:$dest';
    _models[key] = ManagedModel(
      id: key,
      name: p.basenameWithoutExtension(name),
      localPath: dest,
      status: ModelStatus.importing,
      imported: true,
    );
    notifyListeners();
    final tmp = File('$dest.importing');
    try {
      await File(picked).copy(tmp.path);
      await tmp.rename(dest);
    } catch (e) {
      _models.remove(key);
      try {
        if (tmp.existsSync()) await tmp.delete(); // our own partial copy
      } catch (_) {}
      notifyListeners();
      return 'Import failed: $e';
    }
    _models.remove(key);
    _indexModels(); // the catalog claims the file if it is one of its models
    notifyListeners();
    return 'Imported $name';
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
        ramBudgetBytes: ramBudgetMiB == 0 ? SipllmParams.ramBudgetAuto : ramBudgetMiB * 1024 * 1024,
        threads: effectiveThreads,
        maxCtx: maxCtx,
        fastQuant: fastQuant,
        streamLmHead: streamLmHead,
        schedulePolicy: schedulePolicy,
      );

  /// Load [model], unloading the current one first. Throws if the engine
  /// cannot open it. A call while another load is in progress is ignored.
  Future<void> loadModel(ManagedModel model) async {
    if (!model.isInstalled || loading) return;
    loading = true;
    notifyListeners();
    try {
      await unloadModel();
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

  /// Free the loaded model. A chat reply being streamed stops and keeps what
  /// it had. Returns only once the engine has released the model (also when
  /// another unload started it), so a load that follows never holds two
  /// models at once.
  Future<void> unloadModel() async {
    final rt = _rt;
    final sub = _genSub;
    _genEpoch++;
    _genSub = null;
    _endInflightReply(rt == null ? null : '[stopped: the model was unloaded]');
    generating = false;
    _rt = null;
    loadedModel = null;
    if (rt == null) {
      await sub?.cancel();
      notifyListeners();
      final pending = _closing;
      if (pending != null) await pending;
      return;
    }
    unloading = true;
    notifyListeners();
    final closing = () async {
      try {
        await sub?.cancel();
      } catch (_) {}
      await rt.close();
    }();
    _closing = closing;
    try {
      await closing;
    } finally {
      if (identical(_closing, closing)) _closing = null;
      unloading = false;
      notifyListeners();
    }
  }

  // Close the chat turn being streamed: it stops growing, keeps its text plus
  // [note], is saved, and its send() returns.
  void _endInflightReply(String? note) {
    final m = _inflightReply;
    final done = _inflightDone;
    _inflightReply = null;
    _inflightDone = null;
    if (m != null && m.streaming) {
      m.streaming = false;
      if (note != null) m.text = m.text.isEmpty ? note : '${m.text}\n\n$note';
      unawaited(_saveHistory());
    }
    if (done != null && !done.isCompleted) done.complete();
  }

  // ---- chat ----------------------------------------------------------------
  PromptTemplate _template = PromptTemplate.chatml;

  Future<void> send(String text) async {
    final rt = _rt;
    if (rt == null || generating || text.trim().isEmpty) return;
    final epoch = ++_genEpoch;

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
      // The model was unloaded while memory was searched: nothing to send to.
      if (epoch != _genEpoch) return;
    }

    final prompt = _template.build(system, _fitHistory(system));
    messages.add(assistant);
    final completer = Completer<void>();
    _inflightReply = assistant;
    _inflightDone = completer;
    notifyListeners();

    rt.reset();
    final sampler = SipllmSampler(
      temperature: temperature,
      topP: topP,
      topK: 40,
      repeatPenalty: 1.1,
    );

    // Ends this turn; false when unloadModel already ended it.
    bool endTurn() {
      if (!identical(_inflightReply, assistant)) return false;
      _inflightReply = null;
      _inflightDone = null;
      if (epoch == _genEpoch) {
        generating = false;
        _genSub = null;
      }
      return true;
    }

    _genSub = rt.generate(prompt, maxNew: maxTokens, sampler: sampler).listen(
      (tok) {
        assistant.text += tok.piece;
        notifyListeners();
      },
      onDone: () async {
        if (!endTurn()) return;
        assistant
          ..streaming = false
          ..stats = rt.lastStats;
        if (useRag && _memory != null && assistant.text.trim().isNotEmpty) {
          try {
            final e = await rt.embed(assistant.text);
            _memory!.add(text: assistant.text.trim(), embedding: e);
          } catch (_) {}
        }
        notifyListeners();
        unawaited(_saveHistory());
        if (!completer.isCompleted) completer.complete();
      },
      onError: (Object e) {
        if (!endTurn()) return;
        assistant
          ..text += '\n\n[error: $e]'
          ..streaming = false;
        notifyListeners();
        unawaited(_saveHistory());
        if (!completer.isCompleted) completer.complete();
      },
    );
    return completer.future;
  }

  /// The newest turns that fit the context with room left for the reply. The
  /// app has no tokenizer, so it budgets ~3 characters per token, which leaves
  /// margin for English and code. Without this, a history that survives
  /// restarts would eventually overflow the context on every send.
  List<ChatMessage> _fitHistory(String system) {
    final ctx = maxCtx > 0 ? maxCtx : 4096;
    var budget = (ctx - maxTokens - 64) * 3 - system.length;
    final kept = <ChatMessage>[];
    for (final m in messages.reversed) {
      budget -= m.text.length + 16;   // + turn markers
      if (budget < 0 && kept.isNotEmpty) break;
      kept.add(m);
    }
    return kept.reversed.toList();
  }

  void cancelGeneration() {
    if (benchmarking) _benchCancelled = true; // a cut-short run is not a result
    _rt?.cancel();
  }

  /// A raw completion of [prompt] (Playground, Arena). [generating] stays true
  /// until the stream is done, fails, or its listener cancels; before, it was
  /// never cleared, which left Chat in Stop mode with Send unreachable. While
  /// another generation runs this fails at once and changes nothing.
  Stream<SipllmToken> generateStream(String prompt, {int? maxNewTokens}) {
    final rt = _rt;
    if (rt == null) return const Stream.empty();
    if (generating) {
      return Stream.error(StateError('another generation is still running'));
    }
    rt.reset();
    final sampler = SipllmSampler(
      temperature: temperature,
      topP: topP,
      topK: 40,
      repeatPenalty: 1.1,
    );
    final epoch = ++_genEpoch;
    generating = true;
    notifyListeners();
    return onStreamEnd(
      rt.generate(prompt, maxNew: maxNewTokens ?? maxTokens, sampler: sampler),
      () {
        if (epoch != _genEpoch) return; // unloaded since, or a newer generation owns the flag
        generating = false;
        notifyListeners();
      },
    );
  }

  void resetChat() {
    _rt?.reset();
    messages.clear();
    notifyListeners();
    unawaited(_saveHistory());
  }

  // ---- chat history (persisted across launches) ---------------------------
  Future<File> _historyFile() async {
    final support = await getApplicationSupportDirectory();
    return File(p.join(support.path, 'chat_history.json'));
  }

  Future<void> _loadHistory() async {
    try {
      final f = await _historyFile();
      if (!f.existsSync()) return;
      final list = jsonDecode(await f.readAsString()) as List;
      messages
        ..clear()
        ..addAll(list.map((e) {
          final m = e as Map<String, dynamic>;
          return ChatMessage(
            text: m['text'] as String,
            isUser: m['user'] as bool,
            timestamp: DateTime.tryParse(m['ts'] as String? ?? ''),
          );
        }));
    } catch (_) {
      // A corrupt history file must not stop the app; start empty.
    }
  }

  Future<void> _saveHistory() async {
    try {
      final f = await _historyFile();
      final data = jsonEncode([
        for (final m in messages)
          if (!m.streaming)
            {'text': m.text, 'user': m.isUser, 'ts': m.timestamp.toIso8601String()}
      ]);
      // Write then rename, so a kill mid-write keeps the previous history.
      final tmp = File('${f.path}.tmp');
      await tmp.writeAsString(data, flush: true);
      await tmp.rename(f.path);
    } catch (_) {}
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
  bool _benchCancelled = false;

  Future<BenchmarkResult?> runBenchmark({
    String prompt = 'Explain, in three sentences, why the sky is blue.',
    int tokens = 64,
  }) async {
    final rt = _rt;
    if (rt == null || generating || benchmarking || agentRunning) return null;
    // A benchmark uses the runtime like a chat reply: Chat shows Stop (which
    // cancels it) instead of a Send that would fail with "runtime is busy".
    final epoch = ++_genEpoch;
    benchmarking = true;
    _benchCancelled = false;
    generating = true;
    notifyListeners();
    try {
      rt.reset();
      await rt
          .generate(prompt, maxNew: tokens, sampler: const SipllmSampler.greedy())
          .drain<void>();
      // Stopped from Chat, or the model was unloaded mid-run: the numbers
      // describe a cut-short run, so nothing is recorded.
      if (_benchCancelled || epoch != _genEpoch) return null;
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
        ramBudgetBytes: ramBudgetMiB == 0 ? SipllmParams.ramBudgetAuto : ramBudgetMiB * 1024 * 1024,
        vulkan: vulkanAvailable ? 'active' : (vulkanCompiled ? 'compiled' : 'off'),
        engineVersion: engineVersion,
        stats: stats,
      );
      benchmarks.insert(0, result);
      return result;
    } finally {
      benchmarking = false;
      if (epoch == _genEpoch) generating = false; // else unloaded meanwhile
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

/// Where catalog model [c] may be on disk, in the order it is looked for: the
/// download's name (`<id>.gguf`), then its Hugging Face file name, in [modelsDir]
/// and then in [legacyDir].
List<String> catalogFilePaths(CatalogModel c, String modelsDir, {String? legacyDir}) {
  final names = {c.localFileName, p.basename(c.file)};
  return [
    for (final dir in [modelsDir, if (legacyDir != null) legacyDir])
      for (final n in names) p.join(dir, n),
  ];
}

/// [source], calling [onEnd] once when it is over: when it is done, on its
/// first error (a runtime's errors end its generation), or when the listener
/// cancels. [onEnd] runs before the listener sees the end.
Stream<T> onStreamEnd<T>(Stream<T> source, void Function() onEnd) {
  var ended = false;
  void end() {
    if (ended) return;
    ended = true;
    onEnd();
  }

  StreamSubscription<T>? sub;
  late final StreamController<T> out;
  out = StreamController<T>(
    onListen: () {
      sub = source.listen(
        out.add,
        onError: (Object e, StackTrace st) {
          end();
          out.addError(e, st);
        },
        onDone: () {
          end();
          out.close();
        },
      );
    },
    onPause: () => sub?.pause(),
    onResume: () => sub?.resume(),
    onCancel: () {
      end();
      return sub?.cancel();
    },
  );
  return out.stream;
}

/// The slow cores pay off only when they are not far behind the fast ones.
/// Measured with the CLI on two phones (2026-10-28, same text output):
/// POCO X6 Pro, slow cores at 66% of the top clock: 4 fast cores decode
/// Qwen2.5-1.5B 2-4x faster than all 8. OnePlus Nord CE 2, slow cores at 83%:
/// all 8 are faster for Qwen2.5-3B (decode +5%, prefill +12% vs 2 fast cores).
/// The cut-off between those two points is a guess fitted on two phones.
const fastCoreClockRatio = 0.75;

/// The fast ("big") cores of a big.LITTLE CPU, when its slowest cores top out
/// at [fastCoreClockRatio] of the fastest core's clock or less: those whose top
/// frequency is above the slowest core's. 0, meaning "let the engine decide"
/// (all cores), otherwise, or when any core's frequency is unknown.
int fastCoreCount(List<int?> maxFreqKHz) {
  if (maxFreqKHz.isEmpty) return 0;
  final known = <int>[];
  for (final f in maxFreqKHz) {
    if (f == null || f <= 0) return 0;
    known.add(f);
  }
  final slowest = known.reduce((a, b) => a < b ? a : b);
  final fastest = known.reduce((a, b) => a > b ? a : b);
  if (slowest > fastest * fastCoreClockRatio) return 0;
  return known.where((f) => f > slowest).length;
}

/// Each CPU's top frequency in kHz (cpufreq's cpuinfo_max_freq), in core
/// order; null for a core whose file cannot be read. Empty when the cores
/// cannot be listed.
List<int?> readCpuMaxFreqs({String root = '/sys/devices/system/cpu'}) {
  final cores = <int>[];
  try {
    for (final e in Directory(root).listSync(followLinks: false)) {
      final m = RegExp(r'^cpu(\d+)$').firstMatch(p.basename(e.path));
      if (m != null) cores.add(int.parse(m[1]!));
    }
  } catch (_) {
    return const [];
  }
  cores.sort();
  int? read(int core) {
    try {
      return int.tryParse(
          File(p.join(root, 'cpu$core', 'cpufreq', 'cpuinfo_max_freq')).readAsStringSync().trim());
    } catch (_) {
      return null;
    }
  }

  return [for (final c in cores) read(c)];
}
