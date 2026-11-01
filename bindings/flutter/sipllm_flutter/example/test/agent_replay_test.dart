// agent_replay_test.dart — Phone mode end to end against a replayed Settings
// app (test/replay/settings_replay.dart), on a real model, off the phone.
//
// It runs what the Agent screen runs: the same prompt, tools, policy, guards
// and verify step (lib/core/device/phone_agent.dart), with the real model
// through the plugin's FFI. Only the phone is replaced. Greedy decoding makes
// each (model, goal) one deterministic run; more goals, not repeats, give more
// samples. --fast numerics differ slightly between the Mac and the phone, so
// a replayed run can diverge from the phone's at a close token.
//
//   SIPLLM_AGENT_MODEL  path to the GGUF (skipped when unset)
//   SIPLLM_REPLAY_OUT   directory for transcripts + summary.json (optional)
//   SIPLLM_REPLAY_GOALS comma-separated goal ids (default: all)
//   SIPLLM_REPLAY_BUDGET_MIB RAM budget (default: automatic, as in the app);
//                       it changes speed only, not the output
//   SIPLLM_REPLAY_LABELS=1  the experimental tap-by-name tools (DeviceTools.byLabel)
//   SIPLLM_REPLAY_PAGES=1   the experimental open_settings_page tool (DeviceTools.byPage)
//   SIPLLM_REPLAY_FORMAT    auto (default, as the app) | compact | qwen: the
//                       tool format passed to SipllmAgent.create; a non-default
//                       one adds "-fmt<name>" to the output directory name
//
// Each run is graded twice: the original lenient graders (Goal.correct) and
// the strict ones (Goal.strict, settings_replay.dart). summary.json also
// counts malformed calls (a final answer that is an unparsed tool call),
// rejected calls (a tool result with ok=false), verify rejections, repeat-
// guard hits, max-step stops, the largest prompt, and whether the agent
// reached a screen showing the answer.
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:local_llm_chat/core/device/device_control.dart';
import 'package:local_llm_chat/core/device/phone_agent.dart';
import 'package:sipllm_flutter/sipllm_flutter.dart';

import 'replay/settings_replay.dart';

void main() {
  final modelPath = Platform.environment['SIPLLM_AGENT_MODEL'] ?? '';
  final outDir = Platform.environment['SIPLLM_REPLAY_OUT'] ?? '';
  final byLabel = Platform.environment['SIPLLM_REPLAY_LABELS'] == '1';
  final byPage = Platform.environment['SIPLLM_REPLAY_PAGES'] == '1';
  final formatName = (Platform.environment['SIPLLM_REPLAY_FORMAT'] ?? '').trim().toLowerCase();
  final format = switch (formatName) {
    'compact' => AgentToolFormat.compact,
    'qwen' => AgentToolFormat.qwen,
    _ => AgentToolFormat.auto,
  };
  final only = (Platform.environment['SIPLLM_REPLAY_GOALS'] ?? '')
      .split(',')
      .map((s) => s.trim())
      .where((s) => s.isNotEmpty)
      .toSet();

  test('the replayed main screens compact to what the phone showed', () {
    final top = compactScreen(homeSnapshot(home['home_top']!));
    expect(top.text.split('\n'), containsAll([
      '[2] list (more below: scroll down to see it)',
      '[13] button "Wi-Fi · ExampleNet"',
      '[18] button "Bluetooth · Off"',
      '[5] button "Flexible windows · Watch animated tutorials to …"',
    ]));
    expect(compactScreen(homeSnapshot(home['home_bottom']!)).text, contains('[21] button "About device"'));
  });

  test('the replay bridge navigates like Settings', () async {
    TestWidgetsFlutterBinding.ensureInitialized();
    final bridge = ReplayBridge();
    final tools = DeviceTools(bridge, const DevicePolicy())..resetRun();
    await bridge.act('launch', arg: settings);
    await tools.read();
    var r = await tools.execute(const AgentToolCall(0, 'scroll', {'direction': 'down'}, '{}'));
    expect(r.text, contains('Sounds & vibration'));
    r = await tools.execute(const AgentToolCall(1, 'scroll', {'direction': 'down'}, '{}'));
    expect(r.text, contains('[21] button "About device"'));
    r = await tools.execute(const AgentToolCall(2, 'tap', {'id': '21'}, '{}'));
    expect(r.text, contains('Android version'));
    expect(tools.screen!.numbers, contains('13'));
    r = await tools.execute(const AgentToolCall(3, 'back', {}, '{}'));
    expect(r.ok, isTrue);
    expect(bridge.currentName, 'home_bottom');
    // Battery is com.oplus.battery on ColorOS, readable as part of Settings.
    r = await tools.execute(const AgentToolCall(4, 'tap', {'id': '11'}, '{}'));
    expect(r.text, contains('100%'));
    await tools.execute(const AgentToolCall(5, 'back', {}, '{}'));
    // Back on the main screen is refused before it reaches the bridge.
    r = await tools.execute(const AgentToolCall(6, 'back', {}, '{}'));
    expect(r.ok, isFalse);
  });

  test('tap by name scrolls to the item, and refuses what the policy refuses', () async {
    TestWidgetsFlutterBinding.ensureInitialized();
    final bridge = ReplayBridge();
    final tools = DeviceTools(bridge, const DevicePolicy(), byLabel: true)..resetRun();
    await bridge.act('launch', arg: settings);
    final first = await tools.read();
    expect(first.text, isNot(contains('[')));
    expect(first.text, contains('button "Bluetooth · Off"'));
    var r = await tools.execute(const AgentToolCall(0, 'tap', {'name': 'About device'}, '{}'));
    expect(r.ok, isTrue, reason: r.text);
    expect(bridge.currentName, 'about');
    expect(r.text, contains('Android version · 13'));
    await tools.execute(const AgentToolCall(1, 'back', {}, '{}'));
    expect(bridge.currentName, 'home_bottom');
    // Display & brightness is above: found by scrolling back up.
    r = await tools.execute(const AgentToolCall(2, 'tap', {'name': 'Display & brightness'}, '{}'));
    expect(bridge.currentName, 'display');
    r = await tools.execute(const AgentToolCall(3, 'tap', {'name': 'Dark mode'}, '{}'));
    expect(r.ok, isFalse);
    expect(r.text, contains('"Dark mode" is a toggle'));
    r = await tools.execute(const AgentToolCall(4, 'tap', {'name': 'Teleport'}, '{}'));
    expect(r.text, contains('no item named "Teleport"'));
  });

  test('open_settings_page opens a page by name and refuses unknown ones', () async {
    TestWidgetsFlutterBinding.ensureInitialized();
    final bridge = ReplayBridge();
    final tools = DeviceTools(bridge, const DevicePolicy(), byPage: true)..resetRun();
    await bridge.act('launch', arg: settings);
    await tools.read();
    expect(tools.tools.first.name, 'open_settings_page');
    var r = await tools.execute(const AgentToolCall(0, 'open_settings_page', {'page': 'About'}, '{}'));
    expect(r.ok, isTrue, reason: r.text);
    expect(r.text, contains('Android version · 13'));
    r = await tools.execute(const AgentToolCall(1, 'open_settings_page', {'page': 'factory reset'}, '{}'));
    expect(r.ok, isFalse);
    expect(r.text, contains('no page "factory_reset"'));
    // Back from an opened page is allowed (it returns to the main list).
    r = await tools.execute(const AgentToolCall(2, 'back', {}, '{}'));
    expect(bridge.currentName, 'home_top');
  });

  test('strict grading: OEM-skin versions, totals and contradictions fail', () {
    Goal g(String id) => goals.firstWhere((g) => g.id == id);
    // Android version: the phone's 3B answer fails strict, passes lenient.
    expect(g('android').correct('OxygenOS 13.0 Official version'), isTrue);
    expect(g('android').strict('OxygenOS 13.0 Official version'), isFalse);
    expect(g('android_b').strict('OxygenOS · 13.0 · Official version'), isFalse);
    expect(g('android').strict('The Android version running on this phone is 13.0.'), isTrue);
    expect(g('android').strict('This phone has Android version 13.'), isTrue);
    expect(g('android').strict('Android 13 (OxygenOS 13.0)'), isTrue);
    expect(g('android').strict('Android 12'), isFalse);
    // Storage: used, not total.
    expect(g('storage').strict('128 GB'), isFalse);
    expect(g('storage').strict('The storage usage is shown as 119 GB / 128 GB.'), isTrue);
    expect(g('storage').strict('120 GB of 128 GB is used'), isTrue);
    // Dark mode: the lenient grader fails a right answer that mentions "off".
    expect(g('dark').correct('Light mode is off, dark mode is on.'), isFalse);
    expect(g('dark').strict('Light mode is off, dark mode is on.'), isTrue);
    expect(g('dark').strict('Dark mode is off.'), isFalse);
    expect(g('dark').strict('No, dark mode is not on.'), isFalse);
    expect(g('dark').strict('Yes'), isTrue);
    // Timeout: a brightness value is not the timeout.
    expect(g('timeout').strict('970 seconds'), isFalse);
    expect(g('timeout').strict('Auto screen off is set to 30 minutes.'), isTrue);
    expect(g('ram').strict('8.00 GB'), isTrue);
    expect(g('bluetooth').strict('Bluetooth is on.'), isFalse);
    // An unparsed tool call is not an answer.
    expect(looksLikeToolCall('<tool_call>\n{"name": "tap", "arguments": {"id": 13}}\n</tool_call>'), isTrue);
    expect(looksLikeToolCall('[{"name": "open_settings_page", "arguments": {"page": "about"}}]'), isTrue);
    expect(looksLikeToolCall('{"name": "tap", "parameters": {"id": "13"}}'), isTrue);
    expect(looksLikeToolCall('<start_function_call>call:tap{id:13}<end_function_call>'), isTrue);
    expect(looksLikeToolCall('The battery level is 100%.'), isFalse);
    expect(looksLikeToolCall('Android version · 13'), isFalse);
  });

  test('captured pages compact to readable screens', () {
    for (final name in captured.keys) {
      final bridge = ReplayBridge();
      final s = compactScreen(jsonDecode(File('test/replay/pages/$name.json').readAsStringSync()) as Map<String, Object?>);
      expect(s.elements, isNotEmpty, reason: name);
      expect(bridge, isNotNull);
      // ignore: avoid_print
      if (Platform.environment['SIPLLM_REPLAY_PRINT'] == '1') print('== $name\n${s.text}');
    }
  });

  test('Phone-mode goals on a real model', () async {
    if (modelPath.isEmpty || !File(modelPath).existsSync()) {
      return markTestSkipped('set SIPLLM_AGENT_MODEL to a tool-calling GGUF');
    }
    TestWidgetsFlutterBinding.ensureInitialized();
    final budgetMiB = int.tryParse(Platform.environment['SIPLLM_REPLAY_BUDGET_MIB'] ?? '') ?? 0;
    final rt = await SipllmRuntime.open(modelPath,
        params: SipllmParams(
            ramBudgetBytes: budgetMiB > 0 ? budgetMiB << 20 : SipllmParams.ramBudgetAuto,
            maxCtx: 4096,
            fastQuant: true,
            schedulePolicy: SchedulePolicy.fixed16));
    addTearDown(rt.close);
    final results = <ReplayResult>[];
    final stem = modelPath.split('/').last.replaceAll('.gguf', '') + (byLabel ? '-labels' : '') + (byPage ? '-pages' : '') +
        (format == AgentToolFormat.auto ? '' : '-fmt${format.name}');
    final dir = outDir.isEmpty ? null : (Directory('$outDir/$stem')..createSync(recursive: true));

    for (final g in goals.where((g) => only.isEmpty || only.contains(g.id))) {
      final bridge = ReplayBridge();
      final device = DeviceTools(bridge, const DevicePolicy(), byLabel: byLabel, byPage: byPage)..resetRun();
      await bridge.act('launch', arg: settings);
      final first = await device.read();
      final agent = await SipllmAgent.create(rt,
          tools: device.tools, maxSteps: phoneMaxSteps, maxNewTokens: phoneMaxNewTokens, format: format,
          systemPreamble: byPage
              ? phonePagePreamble
              : byLabel
                  ? phoneLabelSystemPreamble(const DevicePolicy())
                  : phoneSystemPreamble(const DevicePolicy()));
      final log = StringBuffer('goal: ${g.text}\n[first screen]\n${first.text}\n');
      final visited = <String>[bridge.currentName];
      final calls = <String>[];
      final callResults = <Map<String, Object?>>[];
      final finals = <String>[]; // every final answer, including ones the verify step rejected
      var verifyRejects = 0;
      var maxPrompt = 0;
      var firstPrompt = 0;
      final clock = Stopwatch()..start();
      var answer = '';
      var stop = 'none';
      var steps = 0;
      await for (final e in phoneAttempts(agent, g.text, first, device, device.execute, onReject: (a, unseen) {
        verifyRejects++;
        finals.add(a);
        log.writeln('\n[verify] rejected "$a": ${unseen.join(', ')}');
      })) {
        final t = (clock.elapsedMilliseconds / 1000).toStringAsFixed(1);
        final st = rt.lastStats;
        if (st != null) {
          if (firstPrompt == 0) firstPrompt = st.promptTokens;
          if (st.promptTokens > maxPrompt) maxPrompt = st.promptTokens;
        }
        switch (e) {
          case AgentTokenEvent(:final piece):
            log.write(piece);
          case AgentToolCallEvent(:final call):
            calls.add(call.rawJson);
            log.writeln('\n[$t s] [call] ${call.rawJson}'
                '${st == null ? '' : '  (prompt_tokens=${st.promptTokens} ttft=${st.ttftSeconds.toStringAsFixed(2)}s)'}');
          case AgentToolResultEvent(:final call, :final result):
            if (visited.last != bridge.currentName) visited.add(bridge.currentName);
            callResults.add({
              'call': call.rawJson,
              'ok': result.ok,
              if (!result.ok) 'error': result.text.split('\n').first,
            });
            log.writeln('[$t s] [result ok=${result.ok}]\n${result.text}\n[model]');
          case AgentFinalEvent(:final text, steps: final n):
            answer = text.trim();
            finals.add(answer);
            stop = 'final';
            steps = n;
            log.writeln('\n[$t s] [final] steps=$n');
          case AgentStoppedEvent(:final reason, steps: final n, :final message):
            stop = reason.name;
            steps = n;
            log.writeln('\n[$t s] [stopped] ${reason.name} ${message ?? ''}');
        }
      }
      clock.stop();
      agent.dispose();
      final ungrounded = device.ungroundedNumbers(answer);
      final ok = stop == 'final' && ungrounded.isEmpty && g.correct(answer);
      final malformed = finals.where(looksLikeToolCall).length;
      final strictOk = stop == 'final' && ungrounded.isEmpty && !looksLikeToolCall(answer) && g.strict(answer);
      final reached = visited.any(g.pages.contains);
      // Why a strict failure failed: format (an unparsed call), navigation (no
      // screen showing the answer was reached), reading (it was).
      final failure = strictOk
          ? ''
          : malformed > 0 && (stop != 'final' || looksLikeToolCall(answer))
              ? 'format'
              : !reached
                  ? 'navigation'
                  : 'reading';
      final res = ReplayResult(g.id, answer, ok, steps, calls, clock.elapsedMilliseconds / 1000, stop, visited,
          rt.lastStats?.promptTokens ?? 0,
          strict: strictOk,
          extra: {
            'malformed': malformed,
            'rejected_calls': callResults.where((c) => c['ok'] != true).length,
            'repeat_guard': callResults.where((c) => '${c['error']}'.contains('was already done twice')).length,
            'verify_rejects': verifyRejects,
            'max_steps': stop == 'maxSteps',
            'answer_page_reached': reached,
            'failure': failure,
            'first_prompt_tokens': firstPrompt,
            'max_prompt_tokens': maxPrompt,
            'ungrounded': ungrounded.toList(),
            'finals': finals,
            'call_results': callResults,
          });
      results.add(res);
      // ignore: avoid_print
      print('[${ok ? 'PASS' : 'FAIL'}|${strictOk ? 'PASS' : 'FAIL'}] ${g.id}: ${stop == 'final' ? '"$answer"' : stop} '
          '(${calls.length} calls, ${res.seconds.toStringAsFixed(1)} s, screens ${visited.join(' > ')}'
          '${failure.isEmpty ? '' : ', $failure'})');
      if (dir != null) File('${dir.path}/${g.id}.txt').writeAsStringSync(log.toString());
    }
    final pass = results.where((r) => r.correct).length;
    final strictPass = results.where((r) => r.strict).length;
    // ignore: avoid_print
    print('== $stem: $pass/${results.length} correct (lenient), $strictPass/${results.length} strict');
    if (dir != null) {
      File('${dir.path}/summary.json').writeAsStringSync(const JsonEncoder.withIndent('  ').convert({
        'model': modelPath,
        'format': format.name,
        'pages': byPage,
        'labels': byLabel,
        'pass': pass,
        'strict_pass': strictPass,
        'n': results.length,
        'runs': [for (final r in results) r.toJson()]
      }));
    }
  }, timeout: Timeout.none);
}
