// app_fixes_batch2_test.dart — catalog files under either name, imports that
// never overwrite, the agent-run busy flag, and Chat's controls and chips.
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:local_llm_chat/core/controller/sipllm_controller.dart';
import 'package:local_llm_chat/core/models/managed_model.dart';
import 'package:local_llm_chat/core/models/model_catalog.dart';
import 'package:local_llm_chat/features/chat/widgets/chat_input.dart';
import 'package:local_llm_chat/features/chat/widgets/stats_bar.dart';
import 'package:local_llm_chat/features/models/widgets/model_tile.dart';
import 'package:path/path.dart' as p;
import 'package:provider/provider.dart';
import 'package:sipllm_flutter/sipllm_flutter.dart';

// A picker that "picks" [path].
class _FakePicker extends FilePicker {
  _FakePicker(this.path);
  final String path;

  @override
  Future<FilePickerResult?> pickFiles({
    String? dialogTitle,
    String? initialDirectory,
    FileType type = FileType.any,
    List<String>? allowedExtensions,
    Function(FilePickerStatus)? onFileLoading,
    bool allowCompression = true,
    int compressionQuality = 30,
    bool allowMultiple = false,
    bool withData = false,
    bool withReadStream = false,
    bool lockParentWindow = false,
    bool readSequential = false,
  }) async =>
      FilePickerResult([PlatformFile(path: path, name: p.basename(path), size: File(path).lengthSync())]);
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late Directory dir;
  late SipllmController c;
  setUp(() {
    dir = Directory.systemTemp.createTempSync('models_');
    c = SipllmController()..debugUseModelDirs(dir.path);
  });
  tearDown(() {
    c.dispose();
    dir.deleteSync(recursive: true);
  });

  ManagedModel? byId(String id) => c.models.where((m) => m.id == id).firstOrNull;
  final cat = kModelCatalog.first; // smollm2-135m-q8 / SmolLM2-135M-Instruct-Q8_0.gguf
  final hfName = p.basename(cat.file);

  group('a catalog model under either file name (issue 3)', () {
    test('its Hugging Face file name is recognised as the catalog model, not a second model', () async {
      File(p.join(dir.path, hfName)).writeAsBytesSync([1, 2, 3, 4]);
      await c.rescanModels();
      final m = byId(cat.id)!;
      expect(m.status, ModelStatus.installed);
      expect(m.localPath, p.join(dir.path, hfName));
      expect(c.models.where((x) => x.imported), isEmpty);
    });

    test('both names present: one entry each, the copy is marked, no file is deleted', () async {
      final a = File(p.join(dir.path, cat.localFileName))..writeAsBytesSync([1, 2, 3, 4]);
      final b = File(p.join(dir.path, hfName))..writeAsBytesSync([1, 2, 3, 4]);
      await c.rescanModels();
      expect(byId(cat.id)!.localPath, a.path);
      final copy = c.models.firstWhere((x) => x.imported);
      expect(copy.localPath, b.path);
      expect(copy.duplicateOf, cat.name);
      expect(a.existsSync() && b.existsSync(), isTrue);
    });

    test('when the first copy goes, the second becomes the catalog model (listed once)', () async {
      final a = File(p.join(dir.path, cat.localFileName))..writeAsBytesSync([1, 2, 3, 4]);
      final b = File(p.join(dir.path, hfName))..writeAsBytesSync([1, 2, 3, 4]);
      await c.rescanModels();
      a.deleteSync(); // deleted outside the app
      await c.rescanModels();
      expect(byId(cat.id)!.localPath, b.path);
      expect(byId(cat.id)!.status, ModelStatus.installed);
      expect(c.models.where((x) => x.localPath == b.path), hasLength(1));
    });

    test('catalogFilePaths: download name first, then the HF name, then the legacy folder', () {
      expect(catalogFilePaths(cat, '/m', legacyDir: '/l'), [
        '/m/${cat.localFileName}',
        '/m/$hfName',
        '/l/${cat.localFileName}',
        '/l/$hfName',
      ]);
    });
  });

  group('import never overwrites', () {
    test('a file whose name is already in the folder is refused and left as it was', () async {
      final existing = File(p.join(dir.path, 'mine.gguf'))..writeAsBytesSync([9, 9, 9]);
      await c.rescanModels();
      c.loadedModel = byId('mine');
      final src = Directory.systemTemp.createTempSync('pick_');
      addTearDown(() => src.deleteSync(recursive: true));
      final picked = File(p.join(src.path, 'mine.gguf'))..writeAsBytesSync([1]);
      FilePicker.platform = _FakePicker(picked.path);
      final msg = await c.importModel();
      expect(msg, contains('already in the models folder'));
      expect(existing.readAsBytesSync(), [9, 9, 9]);
      expect(identical(c.loadedModel, byId('mine')), isTrue);
    });

    test('a new file is copied in and listed; a catalog file is claimed by its entry', () async {
      final src = Directory.systemTemp.createTempSync('pick_');
      addTearDown(() => src.deleteSync(recursive: true));
      await c.rescanModels();
      FilePicker.platform = _FakePicker((File(p.join(src.path, 'new.gguf'))..writeAsBytesSync([1, 2])).path);
      expect(await c.importModel(), 'Imported new.gguf');
      expect(byId('new')?.status, ModelStatus.installed);
      expect(File(p.join(dir.path, 'new.gguf.importing')).existsSync(), isFalse);

      FilePicker.platform = _FakePicker((File(p.join(src.path, hfName))..writeAsBytesSync([1, 2, 3])).path);
      await c.importModel();
      expect(byId(cat.id)?.status, ModelStatus.installed);
      expect(c.models.where((x) => x.localPath.endsWith(hfName)), hasLength(1));
    });
  });

  test('an agent run marks the runtime in use and notifies', () {
    var notified = 0;
    c.addListener(() => notified++);
    expect(c.runtimeInUse, isFalse);
    c.setAgentRunning(true);
    c.setAgentRunning(true);
    expect(c.runtimeInUse, isTrue);
    expect(notified, 1);
    c.setAgentRunning(false);
    expect(c.runtimeInUse, isFalse);
  });

  testWidgets('Unload during an agent run asks first, and Cancel keeps the model', (tester) async {
    File(p.join(dir.path, 'tiny.gguf')).writeAsBytesSync([1]);
    await c.rescanModels();
    final m = byId('tiny')!;
    c
      ..loadedModel = m
      ..setAgentRunning(true);
    await tester.pumpWidget(ChangeNotifierProvider.value(
      value: c,
      child: MaterialApp(home: Scaffold(body: ModelTile(model: m))),
    ));
    await tester.tap(find.text('Unload'));
    await tester.pumpAndSettle();
    expect(find.text('This stops the agent run.'), findsOneWidget);
    await tester.tap(find.text('Cancel'));
    await tester.pumpAndSettle();
    expect(c.loadedModel, same(m));
  });

  testWidgets('Chat Send and Stop have accessibility labels', (tester) async {
    final handle = tester.ensureSemantics();
    Widget input(bool generating) => MaterialApp(
          home: Scaffold(
            body: ChatInput(enabled: true, generating: generating, onSend: (_) {}, onStop: () {}),
          ),
        );
    await tester.pumpWidget(input(false));
    expect(find.bySemanticsLabel('Send message'), findsOneWidget);
    await tester.pumpWidget(input(true));
    expect(find.bySemanticsLabel('Stop generating'), findsOneWidget);
    handle.dispose();
  });

  testWidgets('Chat stats: whole-MB peak RAM and the thread count up front', (tester) async {
    const stats = SipllmStats(
      ttftSeconds: 0.5,
      prefillTokensPerSecond: 50,
      decodeTokensPerSecond: 12.34,
      peakRssBytes: 2282763878, // 2177.0156 MiB
      weightsResidentBytes: 0,
      kvBytes: 0,
      bytesRead: 0,
      prefetchHits: 0,
      prefetchMisses: 0,
      pinnedLayers: 4,
      nLayers: 28,
      promptTokens: 10,
      genTokens: 20,
      ctxUsed: 30,
      ctxMax: 4096,
    );
    await tester.pumpWidget(const MaterialApp(
      home: Scaffold(body: StatsBar(stats: stats, generating: false, threads: 4)),
    ));
    expect(find.text('2177 MB'), findsOneWidget);
    expect(find.textContaining('2177.0'), findsNothing);
    final labels = tester.widgetList<Text>(find.byType(Text)).map((t) => t.data).toList();
    expect(labels.indexOf('4 threads'), 1, reason: 'right after tok/s');
  });
}
