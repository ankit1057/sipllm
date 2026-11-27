// sipllm_agent.dart — Nishachar, the engine's tool-calling agent loop, driven
// from Dart one step at a time.
//
// The engine owns the conversation, the model's chat template, tool-call
// parsing and the step bound. The app owns the tools: every call the model
// makes arrives here as an [AgentToolCall], and [SipllmAgent.run]'s executor
// decides whether and how to run it (a permission prompt, a UI action, a file
// read). Nothing executes inside the engine. Steps are greedy.
part of 'sipllm_runtime.dart';

/// Argument types a tool can declare (mirror SIPLLM_PARAM_*).
enum AgentParamType {
  string(0),
  integer(1),
  number(2),
  boolean(3),
  stringArray(4);

  const AgentParamType(this.code);
  final int code;
}

/// How tools are shown to the model (mirror SIPLLM_TOOL_FORMAT_*).
enum AgentToolFormat {
  /// The model's own tool format when its chat template has one (Qwen2.5),
  /// else [compact].
  auto(0),

  /// SipLLM's compact tool list.
  compact(1),

  /// Qwen2.5's `<tools>` / `<tool_response>` format.
  qwen(2);

  const AgentToolFormat(this.code);
  final int code;
}

class AgentToolParam {
  const AgentToolParam(this.name,
      {this.type = AgentParamType.string,
      this.required = true,
      this.description = ''});
  final String name;
  final AgentParamType type;
  final bool required;
  final String description;
}

class AgentTool {
  const AgentTool(this.name, this.description, [this.params = const []]);
  final String name;
  final String description;
  final List<AgentToolParam> params;
}

/// A call the model made. [arguments] holds the call's top-level keys; string
/// values are unescaped, other JSON values are verbatim. [rawJson] is the
/// model's text and is not guaranteed to be strict JSON.
class AgentToolCall {
  const AgentToolCall(this.step, this.name, this.arguments, this.rawJson);
  final int step;
  final String name;
  final Map<String, String> arguments;
  final String rawJson;
}

/// What a tool produced. [ok] false means the model sees [text] as an error.
class AgentToolResult {
  const AgentToolResult(this.text, {this.ok = true});
  const AgentToolResult.error(this.text) : ok = false;
  final String text;
  final bool ok;
}

typedef AgentToolExecutor = Future<AgentToolResult> Function(AgentToolCall call);

/// Why a run ended without a final answer.
enum AgentStopReason { maxSteps, empty, toolError, cancelled, error }

sealed class AgentEvent {
  const AgentEvent();
}

/// A piece of the model's text as it is generated.
class AgentTokenEvent extends AgentEvent {
  const AgentTokenEvent(this.piece);
  final String piece;
}

/// The model asked for a tool. The executor is called right after.
class AgentToolCallEvent extends AgentEvent {
  const AgentToolCallEvent(this.call);
  final AgentToolCall call;
}

/// A tool actually ran and this is what the model was told. Show these, not
/// the model's own account: a model can say it did something it never called.
class AgentToolResultEvent extends AgentEvent {
  const AgentToolResultEvent(this.call, this.result);
  final AgentToolCall call;
  final AgentToolResult result;
}

/// The run ended with an answer.
class AgentFinalEvent extends AgentEvent {
  const AgentFinalEvent(this.text, this.steps, this.toolCalls);
  final String text;
  final int steps;
  final int toolCalls;
}

/// The run ended without an answer.
class AgentStoppedEvent extends AgentEvent {
  const AgentStoppedEvent(this.reason, this.steps, [this.message]);
  final AgentStopReason reason;
  final int steps;
  final String? message;
}

/// A Nishachar agent bound to one loaded model.
///
/// ```dart
/// final agent = await SipllmAgent.create(rt, tools: const [
///   AgentTool('read_file', 'Read a text file.', [AgentToolParam('path')]),
/// ]);
/// await for (final e in agent.run('What does notes.txt say?', (call) async {
///   return AgentToolResult(await File(call.arguments['path']!).readAsString());
/// })) { ... }
/// await agent.dispose();
/// ```
///
/// While a run is active the runtime is busy: chat [SipllmRuntime.generate]
/// calls are refused until it ends. The agent shares the runtime's KV cache;
/// [SipllmRuntime.reset] before chatting on it again.
class SipllmAgent {
  SipllmAgent._(this._rt, this._id);

  final SipllmRuntime _rt;
  final int _id;
  bool _running = false;
  bool _cancelled = false;
  bool _disposed = false;

  bool get isRunning => _running;

  static Future<SipllmAgent> create(
    SipllmRuntime runtime, {
    required List<AgentTool> tools,
    int maxSteps = 8,
    int maxNewTokens = 256,
    AgentToolFormat format = AgentToolFormat.auto,
    bool stopOnToolError = false,
    String? systemPreamble,
  }) async {
    final reply = await runtime._request({
      'cmd': 'agent_new',
      'maxSteps': maxSteps,
      'maxNewTokens': maxNewTokens,
      'toolFormat': format.code,
      'stopOnToolError': stopOnToolError,
      'systemPreamble': systemPreamble,
      'tools': [
        for (final t in tools)
          [
            t.name,
            t.description,
            [
              for (final p in t.params)
                [p.name, p.type.code, p.required, p.description]
            ]
          ]
      ],
    });
    return SipllmAgent._(runtime, reply['agent'] as int);
  }

  /// Run [goal] to an answer or a stop. Each tool call is passed to [execute];
  /// a throw from it is reported to the model as an error result.
  Stream<AgentEvent> run(String goal, AgentToolExecutor execute) {
    if (_disposed) return Stream.error(SipllmException('agent is disposed'));
    if (_running) return Stream.error(SipllmException('agent is already running'));
    if (_rt._closed) return Stream.error(SipllmException('runtime is closed'));
    if (_rt._busy) return Stream.error(SipllmException('runtime is busy'));
    final out = StreamController<AgentEvent>();
    _running = true;
    _cancelled = false;
    _rt._busy = true;
    () async {
      var steps = 0;
      var calls = 0;
      try {
        await _rt._request({'cmd': 'agent_start', 'agent': _id, 'goal': goal});
        while (true) {
          if (_cancelled) {
            out.add(AgentStoppedEvent(AgentStopReason.cancelled, steps));
            return;
          }
          final step = await _rt._request(
              {'cmd': 'agent_step', 'agent': _id},
              onMessage: (m) {
                if (m['token'] != null) out.add(AgentTokenEvent(m['token'] as String));
              },
              isFinal: (m) => m.containsKey('step') || m.containsKey('error'));
          final r = step['step'] as Map;
          steps = r['steps'] as int;
          _rt._lastStats = SipllmStats.fromMap(r['stats'] as Map<String, Object?>);
          final ev = r['event'] as int;
          switch (ev) {
            case kAgentToolCall:
              break;
            case kAgentFinal:
              out.add(AgentFinalEvent(r['text'] as String, steps, calls));
              return;
            case kAgentCancelled:
              out.add(AgentStoppedEvent(AgentStopReason.cancelled, steps));
              return;
            case kAgentMaxSteps:
              out.add(AgentStoppedEvent(AgentStopReason.maxSteps, steps));
              return;
            case kAgentEmpty:
              out.add(AgentStoppedEvent(AgentStopReason.empty, steps));
              return;
            case kAgentToolError:
              out.add(AgentStoppedEvent(AgentStopReason.toolError, steps));
              return;
            default:
              out.add(AgentStoppedEvent(
                  AgentStopReason.error, steps, r['error'] as String?));
              return;
          }
          final call = AgentToolCall(
            steps,
            r['name'] as String,
            {
              for (final kv in (r['args'] as List))
                (kv as List)[0] as String: kv[1] as String
            },
            r['raw'] as String,
          );
          out.add(AgentToolCallEvent(call));
          AgentToolResult result;
          try {
            result = await execute(call);
          } catch (e) {
            result = AgentToolResult.error('error: $e');
          }
          if (_cancelled) {
            // The tool ran (it cannot be recalled) but the model is not told.
            out.add(AgentToolResultEvent(call, result));
            out.add(AgentStoppedEvent(AgentStopReason.cancelled, steps));
            return;
          }
          out.add(AgentToolResultEvent(call, result));
          ++calls;
          final sub = await _rt._request({
            'cmd': 'agent_submit',
            'agent': _id,
            'result': result.text,
            'ok': result.ok,
          });
          steps = sub['steps'] as int;
          if (sub['submitted'] == 0) {
            out.add(AgentStoppedEvent(
                sub['stop'] == kAgentToolError
                    ? AgentStopReason.toolError
                    : AgentStopReason.maxSteps,
                steps));
            return;
          }
        }
      } catch (e) {
        out.add(AgentStoppedEvent(AgentStopReason.error, steps, '$e'));
      } finally {
        _running = false;
        _rt._busy = false;
        await out.close();
      }
    }();
    return out.stream;
  }

  /// Stop the run: an in-flight step stops at the next token (prefill is not
  /// interruptible), a tool already running finishes but its result is not
  /// sent to the model, and no further step starts.
  void cancel() {
    if (!_running) return;
    _cancelled = true;
    _rt.cancel();
  }

  /// Free the native agent. Cancels a running run first.
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    cancel();
    if (!_rt._closed) _rt._toWorker.send({'cmd': 'agent_free', 'agent': _id});
  }
}
