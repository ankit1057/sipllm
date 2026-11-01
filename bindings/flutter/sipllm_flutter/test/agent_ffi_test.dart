// agent_ffi_test.dart — the Nishachar agent through Dart -> isolate -> C ABI ->
// engine, plus runtime lifecycle.
//
//   SIPLLM_TEST_MODEL   any model (the toy GGUF is enough): lifecycle checks.
//   SIPLLM_AGENT_MODEL  an instruct model that calls tools (Qwen2.5-1.5B-Instruct
//                       Q8_0 is the measured one): the real tool-call run.
// Each group skips when its model is absent.
import 'dart:async';
import 'dart:io';

import 'package:test/test.dart';

import 'package:sipllm_flutter/src/runtime/sipllm_runtime.dart';
import 'package:sipllm_flutter/src/runtime/sipllm_types.dart';

const _calc = AgentTool(
  'calc',
  "Evaluate an integer arithmetic expression '<int> <op> <int>', op one of + - * /.",
  [AgentToolParam('expr', description: "the expression, e.g. '6 * 7'")],
);
const _read = AgentTool(
  'read_file',
  'Read a text file from the working directory.',
  [AgentToolParam('path', description: 'relative path of the file')],
);

Future<AgentToolResult> _execute(AgentToolCall call) async {
  switch (call.name) {
    case 'calc':
      final m = RegExp(r'^\s*(-?\d+)\s*([-+*/])\s*(-?\d+)\s*$')
          .firstMatch(call.arguments['expr'] ?? '');
      if (m == null) return const AgentToolResult.error('error: bad expression');
      final a = int.parse(m[1]!), b = int.parse(m[3]!);
      final r = switch (m[2]) { '+' => a + b, '-' => a - b, '*' => a * b, _ => a ~/ b };
      return AgentToolResult('$r');
    case 'read_file':
      return call.arguments['path'] == 'notes.txt'
          ? const AgentToolResult('The access code is TANGERINE-4711.')
          : const AgentToolResult.error('error: file not found');
  }
  return const AgentToolResult.error('error: unknown tool');
}

void main() {
  final toyPath = Platform.environment['SIPLLM_TEST_MODEL'] ?? '/tmp/sip_toy.gguf';
  final hasToy = File(toyPath).existsSync();
  final agentPath = Platform.environment['SIPLLM_AGENT_MODEL'] ?? '';
  final hasAgentModel = agentPath.isNotEmpty && File(agentPath).existsSync();

  group('lifecycle (any model)', () {
    test('an agent run always ends, and blocks chat while it runs', () async {
      if (!hasToy) return markTestSkipped('no model at $toyPath');
      final rt = await SipllmRuntime.open(toyPath, params: const SipllmParams(threads: 2));
      addTearDown(rt.close);
      final agent = await SipllmAgent.create(rt,
          tools: const [_calc], maxSteps: 2, maxNewTokens: 8);
      final events = <AgentEvent>[];
      final run = agent.run('What is 2 + 2?', _execute);
      // While the run is active the runtime refuses a chat generation.
      await expectLater(rt.generate('hi', maxNew: 1).toList(), throwsA(isA<SipllmException>()));
      await for (final e in run) {
        events.add(e);
      }
      expect(events.last, anyOf(isA<AgentFinalEvent>(), isA<AgentStoppedEvent>()));
      expect(agent.isRunning, isFalse);
      // And chat works again afterwards.
      rt.reset();
      expect(await rt.generate('hi', maxNew: 2, sampler: const SipllmSampler.greedy()).toList(),
          isNotEmpty);
      await agent.dispose();
    }, timeout: const Timeout(Duration(seconds: 60)));

    test('close during a generation frees the worker and ends the stream', () async {
      if (!hasToy) return markTestSkipped('no model at $toyPath');
      final rt = await SipllmRuntime.open(toyPath, params: const SipllmParams(threads: 2));
      final stream = rt.generate('hello', maxNew: 100000, sampler: const SipllmSampler.greedy());
      final done = Completer<void>();
      var tokens = 0;
      stream.listen((_) => tokens++, onError: (_) {}, onDone: done.complete);
      // Close while tokens are still flowing.
      while (tokens < 2) {
        await Future<void>.delayed(const Duration(milliseconds: 5));
      }
      final sw = Stopwatch()..start();
      await rt.close(timeout: const Duration(seconds: 20));
      expect(sw.elapsed, lessThan(const Duration(seconds: 20)),
          reason: 'close waited for the worker instead of timing out');
      await done.future.timeout(const Duration(seconds: 5));
      expect(tokens, lessThan(100000));
      // The runtime refuses work after close.
      await expectLater(rt.generate('x').toList(), throwsA(isA<SipllmException>()));
    }, timeout: const Timeout(Duration(seconds: 60)));
  });

  group('real model', () {
    test('the model calls a tool, the app runs it, the answer uses its result', () async {
      if (!hasAgentModel) return markTestSkipped('set SIPLLM_AGENT_MODEL to an instruct GGUF');
      final rt = await SipllmRuntime.open(agentPath,
          params: const SipllmParams(fastQuant: true));
      addTearDown(rt.close);
      final agent = await SipllmAgent.create(rt, tools: const [_calc, _read]);
      addTearDown(agent.dispose);

      for (final (goal, tool, want) in [
        ('What is 1234 * 5678? Use the calc tool.', 'calc', '7006652'),
        ('Read the file notes.txt with the read_file tool and tell me the access code in it.',
            'read_file', 'TANGERINE-4711'),
      ]) {
        final events = await agent.run(goal, _execute).toList();
        final ran = events.whereType<AgentToolResultEvent>().toList();
        expect(ran, isNotEmpty, reason: 'no tool ran for: $goal');
        expect(ran.first.call.name, tool);
        expect(ran.first.result.ok, isTrue);
        final last = events.last;
        expect(last, isA<AgentFinalEvent>(), reason: '$goal ended with $last');
        expect((last as AgentFinalEvent).text, contains(want));
        expect(last.toolCalls, ran.length);
        expect(events.whereType<AgentTokenEvent>(), isNotEmpty);
      }
    }, timeout: const Timeout(Duration(minutes: 5)));

    test('close during a long prefill waits for the worker and frees the model', () async {
      if (!hasAgentModel) return markTestSkipped('set SIPLLM_AGENT_MODEL to an instruct GGUF');
      final before = ProcessInfo.currentRss;
      final rt = await SipllmRuntime.open(agentPath, params: const SipllmParams(fastQuant: true));
      // Load the weights: a short generation pins the resident layers.
      await rt.generate('hi', maxNew: 1, sampler: const SipllmSampler.greedy()).toList();
      final loaded = ProcessInfo.currentRss;
      // ~1,000-token prompt: the worker is inside prefill when close() arrives.
      final stream = rt.generate(List.filled(1000, 'word').join(' '), maxNew: 4);
      final done = Completer<void>();
      stream.listen((_) {}, onError: (_) {}, onDone: done.complete);
      await Future<void>.delayed(const Duration(milliseconds: 100));
      final sw = Stopwatch()..start();
      await rt.close(timeout: const Duration(seconds: 100));
      sw.stop();
      await done.future.timeout(const Duration(seconds: 5));
      final after = ProcessInfo.currentRss;
      // ignore: avoid_print
      print('rss before=${before >> 20} loaded=${loaded >> 20} after=${after >> 20} MB, close ${sw.elapsedMilliseconds} ms');
      expect(sw.elapsed, lessThan(const Duration(seconds: 100)), reason: 'close timed out instead of the worker confirming');
      expect(loaded - after, greaterThan(500 << 20), reason: 'the model was not freed by close()');
    }, timeout: const Timeout(Duration(minutes: 5)));

    test('cancel stops a run with an explicit cancelled stop', () async {
      if (!hasAgentModel) return markTestSkipped('set SIPLLM_AGENT_MODEL to an instruct GGUF');
      final rt = await SipllmRuntime.open(agentPath,
          params: const SipllmParams(fastQuant: true));
      addTearDown(rt.close);
      final agent = await SipllmAgent.create(rt, tools: const [_calc]);
      addTearDown(agent.dispose);
      final events = <AgentEvent>[];
      await for (final e in agent.run('What is 1234 * 5678? Use the calc tool.', _execute)) {
        events.add(e);
        if (e is AgentTokenEvent) agent.cancel();
      }
      final last = events.last;
      expect(last, isA<AgentStoppedEvent>());
      expect((last as AgentStoppedEvent).reason, AgentStopReason.cancelled);
      expect(events.whereType<AgentToolResultEvent>(), isEmpty);
    }, timeout: const Timeout(Duration(minutes: 5)));
  });
}
