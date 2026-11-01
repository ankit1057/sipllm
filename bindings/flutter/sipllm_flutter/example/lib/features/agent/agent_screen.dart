import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';
import 'package:sipllm_flutter/sipllm_flutter.dart';

import '../../core/controller/sipllm_controller.dart';
import '../../core/device/device_control.dart';
import '../../core/device/phone_agent.dart';
import '../../core/theme/app_theme.dart';

/// Nishachar agent on the loaded model, in two modes:
///  - Files: four harmless tools (integer arithmetic, echo, and listing/reading
///    files inside an app-private workspace);
///  - Phone: the model operates allowlisted apps (Settings) through the
///    accessibility service, reading the screen and tapping, scrolling and
///    going back. Toggles, typing and anything that changes or deletes are
///    refused (device_control.dart). A Stop bar stays on screen over the app.
/// The log shows what each tool was actually asked and returned, separately
/// from the model's own text, because a model can claim a call it never made.
class AgentScreen extends StatefulWidget {
  const AgentScreen({super.key});

  @override
  State<AgentScreen> createState() => _AgentScreenState();
}

enum _Kind { model, call, result, finalAnswer, stopped }

class _Entry {
  _Entry(this.kind, this.text, {this.ok = true});
  final _Kind kind;
  String text;
  final bool ok;
}

// Kept alive: the Hub's TabBarView disposes a tab that scrolls out of view,
// which lost the mode, the page switch and the goal, and (through dispose)
// cancelled a run in progress.
class _AgentScreenState extends State<AgentScreen>
    with WidgetsBindingObserver, AutomaticKeepAliveClientMixin<AgentScreen> {
  @override
  bool get wantKeepAlive => true;

  static const _tools = [
    AgentTool('calc', "Evaluate an integer arithmetic expression '<int> <op> <int>', op one of + - * /.",
        [AgentToolParam('expr', description: "the expression, e.g. '6 * 7'")]),
    AgentTool('echo', 'Return the given text verbatim.',
        [AgentToolParam('text', description: 'the text to echo back')]),
    AgentTool('list_files', 'List the files in the working directory.'),
    AgentTool('read_file', 'Read a text file from the working directory.',
        [AgentToolParam('path', description: 'relative path of the file')]),
  ];
  static const _examples = [
    'What is 1234 * 5678? Use the calc tool.',
    'What port number is set in config.txt?',
    'List the files, then read notes.txt and tell me the access code.',
  ];
  static const _phoneExamples = [
    'What Android version is this phone running?',
    'Is Bluetooth on or off?',
    'How much storage is used on this phone?',
  ];
  final _goal = TextEditingController(text: _examples.first);
  bool _phone = false;
  bool _serviceOn = false;
  late final DeviceBridge _bridge = DeviceBridge(onStop: _stop);
  // Experimental, off by default: the model opens Settings pages by name
  // (open_settings_page) instead of tapping its way to them. Same tools and
  // preamble as the replay's SIPLLM_REPLAY_PAGES=1 runs.
  bool _byPage = false;
  late DeviceTools _device = DeviceTools(_bridge, const DevicePolicy());
  bool _deviceRun = false;
  // Plain-text record of the run (goal, every tool call and result, the
  // model's text, per-step stats) in the app's external files dir
  // (agent_runs/). Rewritten after every event, so a run the OS kills still
  // leaves its record. Phone mode only records allowlisted apps.
  final _transcript = StringBuffer();
  SipllmRuntime? _rtForLog;
  File? _transcriptFile;
  final _scroll = ScrollController();
  final _entries = <_Entry>[];
  SipllmAgent? _agent;
  StreamSubscription<AgentEvent>? _sub;
  Directory? _workspace;
  final _clock = Stopwatch();

  // Set by Stop (the app's or the overlay's). SipllmAgent.cancel() does
  // nothing between runs, so a Stop during setup or during the verify step's
  // re-read would otherwise be lost and the run would carry on.
  bool _stopRequested = false;
  // True from Run until the agent exists, so Run cannot start a second run.
  bool _starting = false;

  bool get _running => _agent != null || _starting;

  // Told when a run starts and ends, so the rest of the app knows the
  // runtime is in use (Chat, Playground, Arena, Benchmark, Load/Unload).
  SipllmController? _ctl;

  void _stop() {
    _stopRequested = true;
    _agent?.cancel();
    if (mounted) setState(() {});
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _prepareWorkspace();
    _checkService();
  }

  // Re-check after the user comes back from Accessibility settings.
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) _checkService();
  }

  Future<void> _checkService() async {
    if (!Platform.isAndroid) return;
    try {
      final on = await _bridge.isEnabled();
      if (mounted && on != _serviceOn) setState(() => _serviceOn = on);
    } catch (_) {}
  }

  String _describe(AgentToolCall call) => switch (call.name) {
        'tap' => () {
            final id = int.tryParse(call.arguments['id'] ?? '');
            final e = id == null ? null : _device.screen?.byId(id);
            return 'tap [${call.arguments['id']}]${e == null ? '' : ' "${e.label}"'}';
          }(),
        'scroll' => 'scroll ${call.arguments['direction'] ?? 'down'}',
        'open_app' => 'open ${call.arguments['name']}',
        'open_settings_page' => 'open page ${call.arguments['page']}',
        _ => call.name,
      };

  Future<AgentToolResult> _executePhone(AgentToolCall call) async {
    // call.step counts completed steps, so this call belongs to step + 1.
    await _bridge.setStatus('SipLLM agent · step ${call.step + 1}: ${_describe(call)}');
    final r = await _device.execute(call);
    await _bridge.setStatus('SipLLM agent · step ${call.step + 2}: thinking…');
    return r;
  }

  Future<void> _prepareWorkspace() async {
    try {
      final support = await getApplicationSupportDirectory();
      final dir = Directory(p.join(support.path, 'agent_workspace'));
      if (!dir.existsSync()) {
        await dir.create(recursive: true);
        await File(p.join(dir.path, 'notes.txt')).writeAsString('The access code is TANGERINE-4711.\n');
        await File(p.join(dir.path, 'config.txt')).writeAsString('host=localhost\nport=8731\n');
      }
      if (mounted) setState(() => _workspace = dir);
    } catch (e) {
      if (mounted) {
        setState(() => _entries.add(_Entry(_Kind.stopped, 'Could not prepare the workspace: $e', ok: false)));
      }
    }
  }

  // Resolve a model-supplied relative path inside the workspace, or throw.
  File _confined(String raw) {
    final root = _workspace!.resolveSymbolicLinksSync();
    if (raw.isEmpty || p.isAbsolute(raw) || p.split(raw).contains('..')) {
      throw const FileSystemException('only relative paths inside the workspace are allowed');
    }
    final file = File(p.join(root, raw));
    final real = file.existsSync() ? file.resolveSymbolicLinksSync() : file.path;
    if (!p.isWithin(root, real)) throw const FileSystemException('path escapes the workspace');
    return File(real);
  }

  Future<AgentToolResult> _execute(AgentToolCall call) async {
    switch (call.name) {
      case 'calc':
        final m = RegExp(r'^\s*(-?\d+)\s*([-+*/])\s*(-?\d+)\s*$').firstMatch(call.arguments['expr'] ?? '');
        if (m == null) return const AgentToolResult.error("error: expected '<int> <op> <int>'");
        final a = int.parse(m[1]!), b = int.parse(m[3]!);
        if (m[2] == '/' && b == 0) return const AgentToolResult.error('error: division by zero');
        return AgentToolResult('${switch (m[2]) { '+' => a + b, '-' => a - b, '*' => a * b, _ => a ~/ b }}');
      case 'echo':
        return AgentToolResult(call.arguments['text'] ?? '');
      case 'list_files':
        final names = _workspace!.listSync().whereType<File>().map((f) => p.basename(f.path)).toList()..sort();
        return AgentToolResult(names.isEmpty ? '(empty)' : names.join('\n'));
      case 'read_file':
        final f = _confined(call.arguments['path'] ?? '');
        if (!f.existsSync()) return AgentToolResult.error('error: file not found: ${call.arguments['path']}');
        // A small model's context is ~2k tokens: return at most 4 KB.
        final text = await f.readAsString();
        return AgentToolResult(text.length <= 4096
            ? text
            : '${text.substring(0, 4096)}\n[... truncated: ${text.length} characters in total]');
    }
    return AgentToolResult.error('error: unknown tool ${call.name}');
  }

  Future<void> _run(SipllmRuntime rt) async {
    final goal = _goal.text.trim();
    if (goal.isEmpty || _running || _workspace == null) return;
    FocusScope.of(context).unfocus();
    _stopRequested = false;
    final ctl = _ctl = context.read<SipllmController>();
    setState(() {
      _starting = true;
      _entries.clear();
      _entries.add(_Entry(_Kind.model, ''));
    });
    ctl.setAgentRunning(true);
    try {
      await _start(rt, goal);
    } finally {
      _starting = false;
      if (_agent == null) ctl.setAgentRunning(false); // never started, or already over
      if (mounted) setState(() {});
    }
  }

  Future<void> _start(SipllmRuntime rt, String goal) async {
    _clock
      ..reset()
      ..start();
    _rtForLog = rt;
    _transcript
      ..clear()
      ..writeln('# ${DateTime.now().toIso8601String()}  mode=${_phone ? 'phone' : 'files'}'
          '${_phone ? '  tools=${_byPage ? 'pages' : 'ids'}' : ''}')
      ..writeln('goal: $goal');
    await _openTranscript();
    try {
      final SipllmAgent agent;
      CompactScreen? first;
      AgentToolExecutor execute = _execute;
      if (_phone) {
        if (!await _bridge.isEnabled()) {
          if (mounted) {
            setState(() => _entries.add(_Entry(_Kind.stopped,
                'Turn on "SipLLM agent device control" in Accessibility settings first.', ok: false)));
          }
          _clock.stop();
          return;
        }
        // Open Settings and read its first screen before step 0: one model
        // step (and its prefill) fewer than asking the model to open it.
        _deviceRun = await _bridge.beginRun('SipLLM agent · opening Settings…', const DevicePolicy().packages);
        final launched = await _bridge.act('launch', arg: 'com.android.settings');
        if (launched != 'ok') throw StateError(launched);
        _device
          ..resetRun()
          ..onRawSnapshot = _saveRawSnapshot;
        final screen = first = await _device.read();
        if (mounted) setState(() => _entries.insert(0, _Entry(_Kind.result, screen.text)));
        _transcript.writeln('[first screen]\n${screen.text}');
        _flushTranscript();
        await _bridge.setStatus('SipLLM agent · step 1: thinking…');
        agent = await SipllmAgent.create(rt,
            tools: _device.tools, maxSteps: phoneMaxSteps, maxNewTokens: phoneMaxNewTokens,
            systemPreamble: _byPage ? phonePagePreamble : phoneSystemPreamble(const DevicePolicy()));
        execute = _executePhone;
      } else {
        agent = await SipllmAgent.create(rt, tools: _tools, maxSteps: 6);
      }
      if (_stopRequested || !mounted) {
        // Stopped (the overlay's Stop, or the app's) before the first step.
        agent.dispose();
        _transcript.writeln('\n[stopped] before the first step');
        if (mounted) setState(() => _entries.add(_Entry(_Kind.stopped, 'Stopped before the first step.', ok: false)));
        _clock.stop();
        _flushTranscript();
        await _endDeviceRun();
        return;
      }
      setState(() => _agent = agent);
      final events = first == null
          ? agent.run(goal, execute)
          : phoneAttempts(agent, goal, first, _device, execute,
              onReject: _onReject, cancelled: () => _stopRequested);
      _sub = events.listen(_onEvent, onDone: _finish, onError: (Object e) {
        if (mounted) setState(() => _entries.add(_Entry(_Kind.stopped, 'error: $e', ok: false)));
        _finish();
      });
    } catch (e) {
      if (mounted) setState(() => _entries.add(_Entry(_Kind.stopped, 'could not start: $e', ok: false)));
      _clock.stop();
      await _endDeviceRun();
    }
  }

  Future<void> _onReject(String answer, Set<String> unseen) async {
    _transcript.writeln('\n[verify] rejected: ${unseen.join(', ')} was on no screen; retrying');
    if (mounted) {
      setState(() => _entries.add(_Entry(_Kind.stopped,
          'Rejected "$answer": ${unseen.join(', ')} was not on any screen. Retrying.', ok: false)));
    }
    await _bridge.setStatus('SipLLM agent · checking its answer…');
  }

  // Bring the app back over Settings, then end the device run. In that order:
  // ending the run first removed the overlay (this app's only visible window)
  // and stopped the foreground service, after which Android may refuse a
  // background activity start, leaving Settings in front. Each step runs
  // even if the other fails, and the outcome goes into the transcript.
  Future<void> _endDeviceRun() async {
    if (!_deviceRun) return;
    _deviceRun = false;
    String back;
    try {
      // Bounded: endRun (overlay, foreground service, wake lock) must follow
      // even if the reply never comes.
      back = await _bridge
              .returnToAppChecked()
              .timeout(const Duration(seconds: 5), onTimeout: () => 'timeout') ??
          'no reply';
    } catch (e) {
      back = 'error: $e';
    }
    try {
      await _bridge.endRun();
    } catch (e) {
      _transcript.writeln('[endRun] error: $e');
    }
    _transcript.writeln('[return to app] front after return: $back');
    _flushTranscript();
  }

  void _log(AgentEvent e) {
    final st = _rtForLog?.lastStats;
    final stats = st == null
        ? ''
        : '  (prompt_tokens=${st.promptTokens} gen=${st.genTokens} ttft=${st.ttftSeconds.toStringAsFixed(2)}s'
            ' prefill=${st.prefillTokensPerSecond.toStringAsFixed(1)}tok/s decode=${st.decodeTokensPerSecond.toStringAsFixed(1)}tok/s'
            ' pinned=${st.pinnedLayers}/${st.nLayers} read=${st.bytesRead ~/ (1 << 20)}MiB)';
    final t = (_clock.elapsedMilliseconds / 1000).toStringAsFixed(1);
    switch (e) {
      case AgentTokenEvent(:final piece):
        _transcript.write(piece);
      case AgentToolCallEvent(:final call):
        _transcript.writeln('\n[$t s] [call] ${call.rawJson}$stats');
      case AgentToolResultEvent(:final result):
        _transcript.writeln('[$t s] [result ok=${result.ok}]\n${result.text}\n[model]');
      case AgentFinalEvent(:final steps, :final toolCalls):
        _transcript.writeln('\n[$t s] [final] steps=$steps tool_calls=$toolCalls$stats');
      case AgentStoppedEvent(:final reason, :final steps, :final message):
        _transcript.writeln('\n[$t s] [stopped] ${reason.name} steps=$steps ${message ?? ''}$stats');
    }
  }

  void _saveRawSnapshot(Map<Object?, Object?> raw) {
    final f = _transcriptFile;
    if (f == null) return;
    try {
      File(p.join(f.parent.path, 'last_snapshot.json')).writeAsStringSync(jsonEncode(raw));
    } catch (_) {}
  }

  Future<void> _openTranscript() async {
    _transcriptFile = null;
    try {
      final base = Platform.isAndroid ? await getExternalStorageDirectory() : await getApplicationSupportDirectory();
      if (base == null) return;
      final dir = Directory(p.join(base.path, 'agent_runs'));
      await dir.create(recursive: true);
      final name = 'run-${DateTime.now().toIso8601String().replaceAll(':', '-')}.txt';
      _transcriptFile = File(p.join(dir.path, name));
      _flushTranscript();
    } catch (_) {}
  }

  // Rewrites the whole (small) record synchronously, so it is on disk before
  // the next step starts.
  void _flushTranscript() {
    try {
      _transcriptFile?.writeAsStringSync(_transcript.toString(), flush: true);
    } catch (_) {}
  }

  void _onEvent(AgentEvent e) {
    _log(e);
    if (e is! AgentTokenEvent) _flushTranscript();
    if (!mounted) return;
    setState(() {
      switch (e) {
        case AgentTokenEvent(:final piece):
          if (_entries.isEmpty || _entries.last.kind != _Kind.model) _entries.add(_Entry(_Kind.model, ''));
          _entries.last.text += piece;
        case AgentToolCallEvent(:final call):
          final args = call.arguments.entries.map((kv) => '${kv.key}: ${kv.value}').join(', ');
          _entries.add(_Entry(_Kind.call, '${call.name}($args)'));
        case AgentToolResultEvent(:final result):
          _entries.add(_Entry(_Kind.result, result.text, ok: result.ok));
          _entries.add(_Entry(_Kind.model, ''));
        case AgentFinalEvent(:final text, :final steps, :final toolCalls):
          _entries.removeWhere((x) => x.kind == _Kind.model && x.text.trim().isEmpty);
          _entries.add(_Entry(_Kind.finalAnswer,
              'Done in $steps step${steps == 1 ? '' : 's'}, $toolCalls tool call${toolCalls == 1 ? '' : 's'} ran.'));
          // A figure the agent never saw on any screen is a guess: say so.
          final unseen = _deviceRun ? _device.ungroundedNumbers(text) : const <String>{};
          if (unseen.isNotEmpty) {
            _entries.add(_Entry(_Kind.stopped,
                'Unverified: ${unseen.join(', ')} did not appear on any screen the agent read.', ok: false));
            _transcript.writeln('[ungrounded] ${unseen.join(', ')}');
          }
        case AgentStoppedEvent(:final reason, :final steps, :final message):
          _entries.removeWhere((x) => x.kind == _Kind.model && x.text.trim().isEmpty);
          _entries.add(_Entry(_Kind.stopped,
              'Stopped (${reason.name}) after $steps step${steps == 1 ? '' : 's'}${message == null ? '' : ': $message'}',
              ok: false));
      }
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted && _scroll.hasClients) _scroll.jumpTo(_scroll.position.maxScrollExtent);
    });
  }

  void _finish() {
    _clock.stop();
    _flushTranscript();
    final agent = _agent;
    _sub = null;
    _agent = null;
    if (mounted) setState(() {});
    _ctl?.setAgentRunning(false);
    agent?.dispose();
    _endDeviceRun();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    final ctl = _ctl;
    if (ctl != null && _running) {
      // Not while the tree is being torn down: listeners rebuild.
      Future.microtask(() => ctl.setAgentRunning(false));
    }
    _endDeviceRun();
    _agent?.cancel();
    _sub?.cancel();
    _agent?.dispose();
    _goal.dispose();
    _scroll.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    super.build(context); // AutomaticKeepAliveClientMixin
    final c = context.watch<SipllmController>();
    final rt = c.runtime;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            rt == null
                ? 'Load a model in Models to use the agent. A tool-calling instruct model works best (measured: Qwen2.5-1.5B-Instruct).'
                : _phone
                    ? 'Model: ${c.loadedModel?.name ?? rt.modelInfo.arch}  ·  operates Settings: ${_byPage ? 'open page, ' : ''}tap, scroll, back (no toggles, no typing)'
                    : 'Model: ${c.loadedModel?.name ?? rt.modelInfo.arch}  ·  tools: calc, echo, list_files, read_file',
            style: const TextStyle(color: AppTheme.textSec, fontSize: 12),
          ),
          const SizedBox(height: 6),
          SegmentedButton<bool>(
            segments: const [
              ButtonSegment(value: false, label: Text('Files'), icon: Icon(LucideIcons.folder, size: 14)),
              ButtonSegment(value: true, label: Text('Phone'), icon: Icon(LucideIcons.smartphone, size: 14)),
            ],
            selected: {_phone},
            onSelectionChanged: _running
                ? null
                : (v) {
                    setState(() {
                      _phone = v.first;
                      _goal.text = (_phone ? _phoneExamples : _examples).first;
                    });
                    if (_phone && Platform.isAndroid) _bridge.prepareKeepAlive().catchError((_) => false);
                  },
          ),
          if (_phone)
            Align(
              alignment: Alignment.centerLeft,
              child: TextButton.icon(
                onPressed: () => _bridge.openAppSettings(),
                icon: const Icon(LucideIcons.batteryCharging, size: 14),
                label: const Text('Keep-alive settings (Battery usage → Allow background activity)',
                    style: TextStyle(fontSize: 11)),
              ),
            ),
          if (_phone)
            SwitchListTile(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Open Settings pages directly (experimental)', style: TextStyle(fontSize: 12)),
              value: _byPage,
              onChanged: _running
                  ? null
                  : (v) => setState(() {
                        _byPage = v;
                        _device = DeviceTools(_bridge, const DevicePolicy(), byPage: v);
                      }),
            ),
          if (_phone && !_serviceOn)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: OutlinedButton.icon(
                onPressed: () => _bridge.openSettings(),
                icon: const Icon(LucideIcons.accessibility, size: 16),
                label: const Text('Enable "SipLLM agent device control" in Accessibility'),
              ),
            ),
          if (_workspace != null && !_phone)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text('Workspace: ${_workspace!.path}',
                  style: const TextStyle(color: AppTheme.textSec, fontSize: 11), overflow: TextOverflow.ellipsis),
            ),
          const SizedBox(height: 10),
          TextField(
            controller: _goal,
            minLines: 1,
            maxLines: 3,
            enabled: !_running,
            decoration: const InputDecoration(labelText: 'Goal', border: OutlineInputBorder()),
          ),
          const SizedBox(height: 6),
          // One scrolling row, so the log below keeps most of the screen.
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: [
                for (final ex in _phone ? _phoneExamples : _examples)
                  Padding(
                    padding: const EdgeInsets.only(right: 6),
                    child: ActionChip(
                      label: Text(ex, style: const TextStyle(fontSize: 11)),
                      onPressed: _running ? null : () => _goal.text = ex,
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: rt == null || _running || c.generating || c.benchmarking ? null : () => _run(rt),
                  icon: const Icon(LucideIcons.play, size: 16),
                  label: const Text('Run'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _running && !_stopRequested ? _stop : null,
                  icon: const Icon(LucideIcons.square, size: 16),
                  label: const Text('Stop'),
                ),
              ),
            ],
          ),
          if (_clock.elapsedMilliseconds > 0)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text('${(_clock.elapsedMilliseconds / 1000).toStringAsFixed(1)} s',
                  style: const TextStyle(color: AppTheme.textSec, fontSize: 11)),
            ),
          const SizedBox(height: 8),
          Expanded(
            child: ListView.builder(
              controller: _scroll,
              itemCount: _entries.length,
              itemBuilder: (_, i) => _EntryCard(_entries[i]),
            ),
          ),
        ],
      ),
    );
  }
}

class _EntryCard extends StatelessWidget {
  const _EntryCard(this.e);
  final _Entry e;

  @override
  Widget build(BuildContext context) {
    final (IconData icon, Color color, String label) = switch (e.kind) {
      _Kind.model => (LucideIcons.bot, AppTheme.textSec, 'Model'),
      _Kind.call => (LucideIcons.wrench, AppTheme.accent, 'Tool call'),
      _Kind.result => e.ok
          ? (LucideIcons.checkCircle, AppTheme.primary, 'Tool result')
          : (LucideIcons.alertCircle, Colors.redAccent, 'Tool error'),
      _Kind.finalAnswer => (LucideIcons.flag, AppTheme.primary, 'Finished'),
      _Kind.stopped => (LucideIcons.octagon, Colors.amber, 'Stopped'),
    };
    if (e.kind == _Kind.model && e.text.isEmpty) return const SizedBox.shrink();
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.35)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Icon(icon, size: 14, color: color),
            const SizedBox(width: 6),
            Text(label, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w700)),
          ]),
          const SizedBox(height: 4),
          SelectableText(
            e.text,
            style: TextStyle(
              color: AppTheme.textMain,
              fontSize: 13,
              fontFamily: e.kind == _Kind.model || e.kind == _Kind.result ? 'monospace' : null,
            ),
          ),
        ],
      ),
    );
  }
}
