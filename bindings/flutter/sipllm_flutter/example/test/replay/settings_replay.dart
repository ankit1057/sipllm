// settings_replay.dart — a small Settings app for replaying Phone-mode runs
// off the phone: screens in SipAccessibilityService's snapshot format, and a
// DeviceBridge that taps, scrolls and goes back between them.
//
// Main screen: the IV2201's three scroll positions (ColorOS 13) with the
// element ids the phone's own service produced (run transcripts), the Wi-Fi
// name replaced. Pages below it (About device, Display & brightness, Battery,
// Search, Bluetooth) are uiautomator captures of the same phone converted to
// the snapshot format (pages/*.json; their ids differ from the service's).
// Rows with no captured page open a stand-in page that answers nothing.
import 'dart:convert';
import 'dart:io';

import 'package:local_llm_chat/core/device/device_control.dart';

/// One row of a main-screen position.
class Btn {
  const Btn(this.id, this.title, [this.summary, this.to]);
  final int id;
  final String title;
  final String? summary;
  final String? to; // page this opens; null = a stand-in page; '.' = nothing
}

const settings = 'com.android.settings';

/// A main-screen position: rows inside the list with id 2.
class Home {
  const Home(this.rows, {this.down, this.up});
  final List<Btn> rows;
  final String? down, up;
}

const home = <String, Home>{
  'home_top': Home([
    Btn(5, 'Flexible windows', 'Watch animated tutorials to learn how to use floating windows.'),
    Btn(9, 'Log In to Your OnePlus Account', 'Unlock your exclusive benefits'),
    Btn(13, 'Wi-Fi', 'ExampleNet'),
    Btn(16, 'Mobile network'),
    Btn(18, 'Bluetooth', 'Off', 'bluetooth'),
    Btn(21, 'Connection & sharing'),
    Btn(24, 'Wallpapers & style'),
    Btn(26, 'Home screen & Lock screen'),
    Btn(28, 'Display & brightness', null, 'display'),
    Btn(30, 'Settings', null, '.'),
    Btn(33, 'Search', null, 'search'),
  ], down: 'home_mid'),
  'home_mid': Home([
    Btn(3, 'Display & brightness', null, 'display'),
    Btn(5, 'Sounds & vibration'),
    Btn(7, 'Notification & status bar'),
    Btn(10, 'Apps'),
    Btn(12, 'Password & security'),
    Btn(14, 'Privacy'),
    Btn(16, 'Location'),
    Btn(18, 'Safety & emergency'),
    Btn(20, 'Battery', null, 'battery'),
    Btn(23, 'Special features'),
    Btn(25, 'Digital Wellbeing & parental controls'),
    Btn(27, 'Additional Settings'),
    Btn(31, 'Settings', null, '.'),
    Btn(33, 'Search', null, 'search'),
  ], down: 'home_bottom', up: 'home_top'),
  'home_bottom': Home([
    Btn(3, 'Password & security'),
    Btn(5, 'Privacy'),
    Btn(7, 'Location'),
    Btn(9, 'Safety & emergency'),
    Btn(11, 'Battery', null, 'battery'),
    Btn(14, 'Special features'),
    Btn(16, 'Digital Wellbeing & parental controls'),
    Btn(18, 'Additional Settings'),
    Btn(21, 'About device', null, 'about'),
    Btn(23, 'Users & accounts'),
    Btn(25, 'Google'),
    Btn(27, 'Help & feedback'),
    Btn(29, 'Settings', null, '.'),
    Btn(31, 'Search', null, 'search'),
  ], up: 'home_mid'),
};

/// Captured pages: file, and the pages scrolling leads to.
const captured = <String, ({String? down, String? up})>{
  'about': (down: 'about_s1', up: null),
  'about_s1': (down: null, up: 'about'),
  'display': (down: 'display_s1', up: null),
  'display_s1': (down: null, up: 'display'),
  'battery': (down: null, up: null),
  'search': (down: null, up: null),
  'bluetooth': (down: null, up: null),
};

Map<String, Object?> _node(int i, int p,
        {String text = '', bool click = false, bool scroll = false, bool down = false, bool up = false}) =>
    {'i': i, 'p': p, 'text': text, 'desc': '', 'cls': '', 'click': click, 'edit': false,
     'scroll': scroll, 'check': false, 'checked': false, 'password': false, 'enabled': true,
     if (scroll) 'more_down': down, if (scroll) 'more_up': up};

/// The snapshot the phone's service sent for a main-screen position: row [id]
/// is the clickable node, its title the next node.
Map<Object?, Object?> homeSnapshot(Home h) {
  final nodes = <int, Map<String, Object?>>{
    0: _node(0, -1),
    2: _node(2, 0, scroll: true, down: h.down != null, up: h.up != null),
  };
  var extra = 200;
  for (final r in h.rows) {
    nodes[r.id] = _node(r.id, 2, click: true);
    nodes[r.id + 1] = _node(r.id + 1, r.id, text: r.title);
    if (r.summary != null) nodes[extra] = _node(extra++, r.id, text: r.summary!);
  }
  final ordered = nodes.keys.toList()..sort();
  return {'package': settings, 'nodes': [for (final k in ordered) nodes[k]]};
}

/// A stand-in page: its title and two entries that open pages with nothing
/// on them, so a wrong tap costs steps but cannot go deeper forever.
Map<Object?, Object?> standIn(String title, {bool leaf = false}) => {
      'package': settings,
      'nodes': [
        _node(0, -1),
        _node(1, 0, scroll: true),
        _node(2, 1, text: title),
        if (!leaf) ...[
          _node(3, 1, click: true), _node(4, 3, text: '$title options'),
          _node(5, 1, click: true), _node(6, 5, text: 'More'),
        ] else
          _node(3, 1, text: 'Nothing else on this page'),
      ],
    };

/// The label a tap on [index] acts on: the nearest clickable ancestor's first
/// text, as the service taps the nearest clickable ancestor. null = not tappable.
String? tappedLabel(Map<Object?, Object?> snap, int index) {
  final nodes = {for (final m in (snap['nodes'] as List)) (m as Map)['i'] as int: m};
  var n = nodes[index];
  while (n != null && n['click'] != true) {
    n = nodes[n['p'] as int];
  }
  if (n == null) return null;
  final start = n['i'] as int;
  String? first;
  void visit(int i) {
    final m = nodes[i]!;
    final t = ((m['text'] as String?) ?? '').trim();
    final d = ((m['desc'] as String?) ?? '').trim();
    if (first == null && (t.isNotEmpty || d.isNotEmpty)) first = t.isNotEmpty ? t : d;
    for (final c in nodes.values) {
      if (c['p'] == i) visit(c['i'] as int);
    }
  }
  visit(start);
  return first ?? '';
}

/// What is on screen: a name, its snapshot, where scrolling and taps lead.
class _Page {
  _Page(this.name, this.snap, {this.down, this.up, this.links = const {}, this.leaf = false});
  final String name;
  final Map<Object?, Object?> snap;
  final String? down, up;
  final Map<String, String> links; // row label -> page name or '.'
  final bool leaf;
}

/// Settings as a stack of pages, driven like the real service.
class ReplayBridge extends DeviceBridge {
  ReplayBridge({String pagesDir = 'test/replay/pages'}) : _dir = pagesDir;
  final String _dir;
  final _stack = <_Page>[];
  final log = <String>[];
  bool _launcher = false;

  String get currentName => _launcher ? 'launcher' : _stack.last.name;

  _Page _page(String name) {
    final h = home[name];
    if (h != null) {
      return _Page(name, homeSnapshot(h), down: h.down, up: h.up,
          links: {for (final r in h.rows) if (r.to != null) r.title: r.to!});
    }
    final c = captured[name]!;
    final snap = jsonDecode(File('$_dir/$name.json').readAsStringSync()) as Map<String, Object?>;
    return _Page(name, snap, down: c.down, up: c.up);
  }

  @override
  Future<bool> isEnabled() async => true;
  @override
  Future<bool> beginRun(String status, Set<String> allowed) async => true;
  @override
  Future<void> setStatus(String status) async {}
  @override
  Future<void> endRun() async {}
  @override
  Future<void> returnToApp() async {}

  @override
  Future<Map<Object?, Object?>> snapshot({bool settle = true}) async {
    if (_launcher || _stack.isEmpty) {
      return {'package': 'com.oplus.launcher', 'nodes': const [], 'blocked': true};
    }
    final s = _stack.last.snap;
    // The service reads only the run's allowed packages.
    final pkg = s['package'] as String;
    if (!const DevicePolicy().packages.contains(pkg)) return {'package': pkg, 'nodes': const [], 'blocked': true};
    return s;
  }

  @override
  Future<String> act(String action, {int index = -1, String arg = ''}) async {
    log.add('$action${index >= 0 ? ' $index' : ''}${arg.isEmpty ? '' : ' $arg'}');
    switch (action) {
      case 'launch':
        _launcher = false;
        _stack
          ..clear()
          ..add(_page('home_top'));
        return 'ok';
      case 'home':
        _launcher = true;
        return 'ok';
      case 'open_page':
        // An intent straight to the page, on top of the Settings main screen.
        _launcher = false;
        final name = switch (arg) {
          'about' || 'battery' || 'display' || 'bluetooth' => arg,
          'main' => null,
          _ => '',
        };
        _stack
          ..clear()
          ..add(_page('home_top'));
        if (name == '') {
          _stack.add(_Page('page:$arg', standIn(arg[0].toUpperCase() + arg.substring(1).replaceAll('_', ' '))));
        } else if (name != null) {
          _stack.add(_page(name));
        }
        return 'ok';
      case 'back':
        if (_stack.length <= 1) {
          _launcher = true;
        } else {
          _stack.removeLast();
        }
        return 'ok';
      case 'scroll_forward' || 'scroll_backward':
        final cur = _stack.last;
        final next = action == 'scroll_forward' ? cur.down : cur.up;
        if (next == null) return 'error: cannot scroll further';
        _stack
          ..removeLast()
          ..add(_page(next));
        return 'ok';
      case 'tap':
        final cur = _stack.last;
        final label = tappedLabel(cur.snap, index);
        if (label == null) return 'error: element [$index] is not tappable';
        final to = cur.links[label];
        if (to == '.') return 'ok'; // nothing opens
        if (to != null) {
          _stack.add(_page(to));
        } else if (cur.leaf || cur.links.isEmpty && !home.containsKey(cur.name)) {
          _stack.add(_Page('leaf:$label', standIn(label, leaf: true), leaf: true));
        } else {
          _stack.add(_Page('page:$label', standIn(label), links: const {}));
        }
        return 'ok';
    }
    return 'error: unknown action $action';
  }
}

/// A goal, and how to tell a right answer.
class Goal {
  const Goal(this.id, this.text, this.correct, {this.hops = 0, bool Function(String answer)? strict, this.pages = const {}})
      : _strict = strict;
  final String id;
  final String text;

  /// The original (lenient) grader.
  final bool Function(String answer) correct;
  final bool Function(String answer)? _strict;

  /// The strict grader (see [strictGraders]); the lenient one where the two agree.
  bool Function(String answer) get strict => _strict ?? correct;

  /// Screens on which the answer is visible (ReplayBridge.currentName values).
  final Set<String> pages;
  final int hops; // fewest tool calls that reach the answer
}

bool _has(String a, String pattern) => RegExp(pattern, caseSensitive: false).hasMatch(a);

// ---- Strict grading (fixed 2026-10-28, before the model-evaluation runs) ----
//
// The lenient graders above pass "OxygenOS 13.0 Official version" for the
// Android version (\b13\b matches "13.0") and fail "Light mode is off, dark
// mode is on" (it contains "off"). The strict ones:
//  - android: an OEM-skin version ("OxygenOS 13.0", "ColorOS 13") is not the
//    Android version. Those mentions are removed first; 13 (or 13.0) must
//    still be stated. "The Android version ... is 13.0" counts.
//  - storage: the USED amount (119 GB in this capture, 120 on the phone on a
//    later day); the total (128 GB) alone fails.
//  - dark: dark mode stated on/enabled/yes, and not stated off/not on.
//  - timeout 30 minutes, model Nord CE 2 / IV2201, battery 100%, RAM 8 GB,
//    Bluetooth off: as lenient.
// On top of the grader, a strict pass needs a final answer that is not an
// unparsed tool call (looksLikeToolCall) and states no figure that was on no
// screen (the app's own grounding check), as the lenient grade does.

/// A final answer that is really a tool call the engine did not parse.
bool looksLikeToolCall(String a) => RegExp(
        r'<tool_call>|</tool_call>|<\|tool_call\|?>|<start_function_call>|<\|python_tag\|>|functools\[|'
        r'"name"\s*:|^\s*\[?\s*\{|\bcall:[a-z_]+\{',
        caseSensitive: false)
    .hasMatch(a);

bool strictAndroid(String a) {
  final s = a.replaceAll(
      RegExp(r'\b(oxygen\s*os|color\s*os|h2os|oneplus\s*os)\b[^0-9]{0,24}\d+(\.\d+)*', caseSensitive: false), ' ');
  return _has(s, r'\b13(\.0)?\b');
}

bool strictStorage(String a) => _has(a, r'\b(119|120)(\.\d+)?\s*GB');

bool strictDark(String a) {
  if (_has(a, r'dark\s*mode\W+(is\s+)?(currently\s+)?(turned\s+)?(off|disabled|not)\b')) return false;
  if (_has(a, r'\b(not|isn.t)\s+(turned\s+)?(on|enabled)\b')) return false;
  return _has(a, r'\b(yes|on|enabled)\b');
}

// Truth from the phone itself (adb getprop / dumpsys / settings, 2026-10-27)
// and the captured pages.
final goals = <Goal>[
  Goal('android', 'What Android version is this phone running?', (a) => _has(a, r'\b13\b'),
      hops: 3, strict: strictAndroid, pages: {'about', 'about_s1'}),
  Goal('android_b', 'Which version of Android does this phone have?', (a) => _has(a, r'\b13\b'),
      hops: 3, strict: strictAndroid, pages: {'about', 'about_s1'}),
  Goal('bluetooth', 'Is Bluetooth on or off?', (a) => _has(a, r'\boff\b') && !_has(a, r'\bis on\b'),
      hops: 0, pages: {'home_top', 'bluetooth'}),
  Goal('storage', 'How much storage is used on this phone?', (a) => _has(a, r'\b119\b'),
      hops: 3, strict: strictStorage, pages: {'about'}),
  Goal('model', 'What is the model name of this phone?', (a) => _has(a, r'Nord CE 2|IV2201'),
      hops: 3, pages: {'about'}),
  Goal('battery', 'What is the battery level?', (a) => _has(a, r'100\s*%'), hops: 2, pages: {'battery'}),
  Goal('dark', 'Is dark mode on?', (a) => _has(a, r'\b(yes|on|enabled)\b') && !_has(a, r'\b(off|not)\b'),
      hops: 1, strict: strictDark, pages: {'display', 'display_s1'}),
  Goal('timeout', 'How long does the screen stay on before it turns off automatically?',
      (a) => _has(a, r'30\s*min'), hops: 2, pages: {'display_s1'}),
  Goal('ram', 'How much RAM does this phone have?', (a) => _has(a, r'\b8(\.00)?\s*GB'),
      hops: 3, pages: {'about', 'about_s1'}),
];

/// The run's outcome, for the summary table.
class ReplayResult {
  ReplayResult(this.goal, this.answer, this.correct, this.steps, this.calls, this.seconds, this.stop,
      this.screensVisited, this.promptTokens, {this.strict = false, this.extra = const {}});
  final String goal;
  final String answer;
  final bool correct; // lenient grade
  final bool strict; // strict grade
  final Map<String, Object?> extra; // per-run metrics (agent_replay_test.dart)
  final int steps;
  final List<String> calls;
  final double seconds;
  final String stop; // final / maxSteps / ...
  final List<String> screensVisited;
  final int promptTokens;

  Map<String, Object?> toJson() => {
        'goal': goal, 'correct': correct, 'strict': strict, 'answer': answer, 'steps': steps, 'calls': calls,
        'seconds': seconds, 'stop': stop, 'screens': screensVisited, 'prompt_tokens': promptTokens, ...extra,
      };
}
