// app_state_test.dart — app state fixes that need no model or phone: the
// generation busy flag, the models-folder rescan, Unload on the Models
// screen, the Agent screen surviving a tab switch, Stop between the verify
// step's attempts, and the automatic thread count.
import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:local_llm_chat/core/controller/sipllm_controller.dart';
import 'package:local_llm_chat/core/device/device_control.dart';
import 'package:local_llm_chat/core/device/phone_agent.dart';
import 'package:local_llm_chat/core/models/managed_model.dart';
import 'package:local_llm_chat/core/models/model_catalog.dart';
import 'package:local_llm_chat/features/agent/agent_screen.dart';
import 'package:local_llm_chat/features/models/widgets/model_tile.dart';
import 'package:path/path.dart' as p;
import 'package:provider/provider.dart';
import 'package:sipllm_flutter/sipllm_flutter.dart';

// A Settings screen that never shows the number the fake agent answers with.
class _ScreenBridge extends DeviceBridge {
  @override
  Future<Map<Object?, Object?>> snapshot({bool settle = true}) async => {
        'package': 'com.android.settings',
        'nodes': [
          {'i': 0, 'p': -1, 'text': 'About device', 'click': true},
        ],
      };
}

// An agent whose every run ends with an answer stating a figure no screen
// showed, so the verify step always rejects the first one.
class _GuessingAgent implements SipllmAgent {
  int runs = 0;

  @override
  Stream<AgentEvent> run(String goal, AgentToolExecutor execute) async* {
    runs++;
    yield const AgentTokenEvent('It is 42 GB');
    yield const AgentFinalEvent('It is 42 GB', 1, 0);
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('onStreamEnd (Playground/Arena busy flag)', () {
    test('ends once when the stream is done, before the listener sees done', () async {
      final src = StreamController<int>();
      var ends = 0;
      var endedBeforeDone = false;
      final done = Completer<void>();
      onStreamEnd(src.stream, () => ends++).listen((_) {}, onDone: () {
        endedBeforeDone = ends == 1;
        done.complete();
      });
      src
        ..add(1)
        ..add(2);
      await src.close();
      await done.future;
      expect(ends, 1);
      expect(endedBeforeDone, isTrue);
    });

    test('ends once on an error followed by done', () async {
      final src = StreamController<int>();
      var ends = 0;
      final errors = <Object>[];
      final done = Completer<void>();
      onStreamEnd(src.stream, () => ends++).listen((_) {}, onError: errors.add, onDone: done.complete);
      src.addError(StateError('busy'));
      await src.close();
      await done.future;
      expect(ends, 1);
      expect(errors, hasLength(1));
    });

    test('ends once when the listener cancels, and cancels the source', () async {
      var sourceCancelled = false;
      final src = StreamController<int>(onCancel: () => sourceCancelled = true);
      var ends = 0;
      final sub = onStreamEnd(src.stream, () => ends++).listen((_) {});
      src.add(1);
      await Future<void>.delayed(Duration.zero);
      await sub.cancel();
      expect(ends, 1);
      expect(sourceCancelled, isTrue);
    });

    test('an await-for consumer (Arena) sees the flag cleared when its loop ends', () async {
      var busy = true;
      final src = Stream<int>.fromIterable([1, 2, 3]);
      var n = 0;
      await for (final _ in onStreamEnd(src, () => busy = false)) {
        n++;
      }
      expect(n, 3);
      expect(busy, isFalse);
    });
  });

  group('automatic thread count: fast cores', () {
    test('POCO X6 Pro (4 x 2.2, 3 x 3.2, 1 x 3.35 GHz) -> 4', () {
      expect(fastCoreCount([2200000, 2200000, 2200000, 2200000, 3200000, 3200000, 3200000, 3350000]), 4);
    });
    test('OnePlus Nord CE 2 (6 x 2.0, 2 x 2.4 GHz): slow cores at 83% -> 0 (all cores)', () {
      expect(fastCoreCount([2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2400000, 2400000]), 0);
    });
    test('slow cores exactly at the cut-off use the fast cores; just above it, all cores', () {
      expect(fastCoreCount([750, 750, 1000, 1000]), 2);
      expect(fastCoreCount([751, 751, 1000, 1000]), 0);
    });
    test('every core the same top speed -> 0 (engine default)', () {
      expect(fastCoreCount(List.filled(8, 2400000)), 0);
    });
    test('any unreadable core, or none listed -> 0 (engine default)', () {
      expect(fastCoreCount([2000000, null, 2400000, 2400000]), 0);
      expect(fastCoreCount([2000000, 0, 2400000]), 0);
      expect(fastCoreCount(const []), 0);
    });

    test('readCpuMaxFreqs reads cpuN/cpufreq/cpuinfo_max_freq in core order', () {
      final root = Directory.systemTemp.createTempSync('cpu_');
      addTearDown(() => root.deleteSync(recursive: true));
      void core(int i, String? freq) {
        final d = Directory(p.join(root.path, 'cpu$i', 'cpufreq'))..createSync(recursive: true);
        if (freq != null) File(p.join(d.path, 'cpuinfo_max_freq')).writeAsStringSync(freq);
      }

      core(10, '3350000\n');
      core(0, '2200000\n');
      core(1, 'garbage');
      core(2, null);
      Directory(p.join(root.path, 'cpuidle')).createSync(); // not a core
      File(p.join(root.path, 'possible')).writeAsStringSync('0-10');
      expect(readCpuMaxFreqs(root: root.path), [2200000, null, null, 3350000]);
      expect(readCpuMaxFreqs(root: p.join(root.path, 'missing')), isEmpty);
    });

    test('a user-set thread count wins over the fast cores', () {
      final c = SipllmController()..fastCores = 4;
      addTearDown(c.dispose);
      expect(c.effectiveThreads, 4);
      expect(c.autoThreadsLabel, 'auto: 4 fast cores');
      c.threads = 8;
      expect(c.effectiveThreads, 8);
      expect(c.params.threads, 8);
      c
        ..threads = 0
        ..fastCores = 0;
      expect(c.effectiveThreads, 0);
      expect(c.autoThreadsLabel, 'auto');
    });
  });

  group('models folder rescan', () {
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

    test('a GGUF copied in after launch appears; one deleted outside the app goes', () async {
      await c.rescanModels();
      expect(byId('dropped-in'), isNull);
      final f = File(p.join(dir.path, 'dropped-in.gguf'))..writeAsBytesSync([0x47, 0x47, 0x55, 0x46, 1, 2]);
      await c.rescanModels();
      expect(byId('dropped-in')?.status, ModelStatus.installed);
      expect(byId('dropped-in')?.sizeBytes, 6);
      f.deleteSync();
      await c.rescanModels();
      expect(byId('dropped-in'), isNull);
    });

    test('the loaded model, a download and a paused download are left alone', () async {
      await c.rescanModels();
      final cat = kModelCatalog.first;
      final dl = byId(cat.id)!..status = ModelStatus.downloading;
      // The download's file appears (part-way through concatenation).
      File(p.join(dir.path, cat.localFileName)).writeAsBytesSync([1, 2, 3]);
      await c.rescanModels();
      expect(identical(byId(cat.id), dl), isTrue);
      expect(dl.status, ModelStatus.downloading);
      dl.status = ModelStatus.paused;
      await c.rescanModels();
      expect(identical(byId(cat.id), dl), isTrue);

      final f = File(p.join(dir.path, 'mine.gguf'))..writeAsBytesSync([1]);
      await c.rescanModels();
      final mine = byId('mine')!;
      c.loadedModel = mine;
      f.deleteSync();
      await c.rescanModels();
      expect(identical(byId('mine'), mine), isTrue, reason: 'the loaded model stays listed');
    });

    test('a catalog file that is complete shows as installed after a rescan', () async {
      await c.rescanModels();
      final cat = kModelCatalog.first;
      expect(byId(cat.id)?.status, ModelStatus.available);
      File(p.join(dir.path, cat.localFileName)).writeAsBytesSync([1, 2, 3, 4]);
      await c.rescanModels();
      expect(byId(cat.id)?.status, ModelStatus.installed);
    });
  });

  testWidgets('Models: the loaded model has an Unload button that unloads it', (tester) async {
    final dir = Directory.systemTemp.createTempSync('models_');
    addTearDown(() => dir.deleteSync(recursive: true));
    final c = SipllmController()..debugUseModelDirs(dir.path);
    addTearDown(c.dispose);
    File(p.join(dir.path, 'tiny.gguf')).writeAsBytesSync([1, 2, 3]);
    await c.rescanModels();
    final model = c.models.firstWhere((m) => m.id == 'tiny');
    c.loadedModel = model;

    await tester.pumpWidget(ChangeNotifierProvider.value(
      value: c,
      child: MaterialApp(home: Scaffold(body: ModelTile(model: model))),
    ));
    expect(find.text('Unload'), findsOneWidget);
    await tester.tap(find.text('Unload'));
    await tester.pump();
    await tester.pump();
    expect(c.loadedModel, isNull);
    expect(find.text('Unload'), findsNothing);
    expect(find.byTooltip('Load'), findsOneWidget);
  });

  testWidgets('Agent: mode, page switch and goal survive a Hub tab switch', (tester) async {
    final c = SipllmController();
    addTearDown(c.dispose);
    late TabController tabs;
    await tester.pumpWidget(ChangeNotifierProvider.value(
      value: c,
      child: MaterialApp(
        home: DefaultTabController(
          length: 2,
          child: Builder(builder: (context) {
            tabs = DefaultTabController.of(context);
            return const Scaffold(
              body: TabBarView(children: [AgentScreen(), Center(child: Text('other tab'))]),
            );
          }),
        ),
      ),
    ));
    await tester.pump();

    await tester.tap(find.text('Phone'));
    await tester.pump();
    final pageSwitch = find.widgetWithText(SwitchListTile, 'Open Settings pages directly (experimental)');
    expect(tester.widget<SwitchListTile>(pageSwitch).value, isFalse, reason: 'the page switch defaults to off');
    await tester.tap(pageSwitch);
    await tester.pump();
    await tester.enterText(find.widgetWithText(TextField, 'Goal'), 'Is Wi-Fi on?');
    await tester.pump();
    // A focused text field keeps its tab alive by itself; on the phone the
    // keyboard is usually closed when the user switches tabs.
    FocusManager.instance.primaryFocus?.unfocus();
    await tester.pump();

    tabs.animateTo(1);
    await tester.pumpAndSettle();
    expect(find.text('other tab'), findsOneWidget);
    tabs.animateTo(0);
    await tester.pumpAndSettle();

    expect(find.text('Is Wi-Fi on?'), findsOneWidget);
    expect(tester.widget<SwitchListTile>(pageSwitch).value, isTrue);
    expect(
        tester.widget<SegmentedButton<bool>>(find.byType(SegmentedButton<bool>)).selected, {true});
  });

  group('phoneAttempts: Stop between attempts', () {
    late DeviceTools device;
    late CompactScreen first;
    setUp(() async {
      device = DeviceTools(_ScreenBridge(), const DevicePolicy())..resetRun();
      first = await device.read();
    });

    test('without a cancel hook the rejected answer is retried once, as before', () async {
      final agent = _GuessingAgent();
      final events = await phoneAttempts(agent, 'storage?', first, device, device.execute).toList();
      expect(agent.runs, 2);
      expect(events.last, isA<AgentFinalEvent>());
    });

    test('a Stop during the verify step ends the run instead of retrying', () async {
      final agent = _GuessingAgent();
      var stop = false;
      final events = await phoneAttempts(agent, 'storage?', first, device, device.execute,
          onReject: (_, _) => stop = true, cancelled: () => stop).toList();
      expect(agent.runs, 1);
      expect(events.last, isA<AgentStoppedEvent>());
      expect((events.last as AgentStoppedEvent).reason, AgentStopReason.cancelled);
    });
  });
}
