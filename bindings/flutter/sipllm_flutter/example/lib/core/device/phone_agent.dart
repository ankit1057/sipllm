import 'dart:async';

import 'package:sipllm_flutter/sipllm_flutter.dart';

import 'device_control.dart';

/// What Phone mode tells the model, and the run around it. Shared by the Agent
/// screen and the replay test (test/agent_replay_test.dart), so what the
/// replay measures is what the phone runs.
const phonePreamble =
    'You are Nishachar, operating an Android phone for the user through tools. '
    'Each tool result shows the screen after the action: a list of elements, each with an [id]. '
    'Use only ids from the latest screen. Make exactly one tool call per turn. '
    'To find something, tap the section that should contain it, or scroll down when the list says there is more below. '
    'Answer only with what a screen actually showed. Never guess. '
    'When the answer is on the screen, reply with it and no tool call.';

const phoneMaxSteps = 10;
const phoneMaxNewTokens = 160;

String phoneSystemPreamble(DevicePolicy policy) => '$phonePreamble ${policy.hintText}';

/// The preamble for DeviceTools.byLabel (experimental): elements by name.
const phoneLabelPreamble =
    'You are Nishachar, operating an Android phone for the user through tools. '
    'Each tool result shows the screen after the action: the items on it, by name. '
    'To open an item, tap it by its name; tap finds it even if it is further down the list. '
    'Make exactly one tool call per turn. '
    'Answer only with what a screen actually showed. Never guess. '
    'When the answer is on the screen, reply with it and no tool call.';

String phoneLabelSystemPreamble(DevicePolicy policy) => '$phoneLabelPreamble ${policy.labelHintText}';

/// The preamble for DeviceTools.byPage (experimental): open pages by name.
const phonePagePreamble =
    'You are Nishachar, operating an Android phone for the user through tools. '
    'To find something in Settings, first open the page that should contain it with open_settings_page. '
    'Each tool result shows the screen after the action: a list of elements, each with an [id]. '
    'If the page says there is more below, scroll down. Make exactly one tool call per turn. '
    'Answer only with what a screen actually showed. Never guess. '
    'When the answer is on the screen, reply with it and no tool call.';

/// The first user turn: the goal plus the screen the run starts on.
String phoneGoal(String goal, CompactScreen first) => '$goal\n\nThe phone is showing:\n${first.text}';

/// The run, plus Nishachar's verify step: a final answer stating a figure that
/// appeared on no screen the agent read is rejected once, and the agent
/// continues from the current screen with that feedback. A second ungrounded
/// answer stands (the caller flags it as unverified).
///
/// [cancelled] is checked between the attempts: a Stop pressed while the
/// answer is checked and the screen re-read (when no agent run is active, so
/// SipllmAgent.cancel() does nothing) ends with a cancelled stop instead of a
/// retry. Without it the behaviour is unchanged.
Stream<AgentEvent> phoneAttempts(
  SipllmAgent agent,
  String goal,
  CompactScreen first,
  DeviceTools device,
  AgentToolExecutor execute, {
  FutureOr<void> Function(String answer, Set<String> unseen)? onReject,
  bool Function()? cancelled,
}) async* {
  var prompt = phoneGoal(goal, first);
  for (var attempt = 0;; attempt++) {
    AgentFinalEvent? last;
    await for (final e in agent.run(prompt, execute)) {
      if (e is AgentFinalEvent) {
        last = e;
      } else {
        yield e;
      }
    }
    if (last == null) return; // stopped: the stopped event was yielded
    final unseen = device.ungroundedNumbers(last.text);
    if (unseen.isEmpty || attempt >= 1) {
      yield last;
      return;
    }
    await onReject?.call(last.text.trim(), unseen);
    if (cancelled?.call() ?? false) {
      yield AgentStoppedEvent(AgentStopReason.cancelled, last.steps);
      return;
    }
    final screen = await device.read();
    if (cancelled?.call() ?? false) {
      yield AgentStoppedEvent(AgentStopReason.cancelled, last.steps);
      return;
    }
    prompt = '$goal\n\nYour previous answer "${last.text.trim()}" stated ${unseen.join(', ')}, which was not '
        'on any screen you read, so it was a guess. Do not guess. Tap the item that should contain the answer, '
        'then answer from what that screen shows.\n\nThe phone is showing:\n${screen.text}';
  }
}
