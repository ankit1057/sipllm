import 'dart:async';

import 'package:flutter/services.dart';
import 'package:sipllm_flutter/sipllm_flutter.dart';

/// One node of an accessibility snapshot (SipAccessibilityService.kt).
class ScreenNode {
  ScreenNode({
    required this.index,
    required this.parent,
    this.text = '',
    this.desc = '',
    this.cls = '',
    this.clickable = false,
    this.editable = false,
    this.scrollable = false,
    this.checkable = false,
    this.checked = false,
    this.password = false,
    this.enabled = true,
    this.moreDown = false,
    this.moreUp = false,
  });

  factory ScreenNode.fromMap(Map<Object?, Object?> m) => ScreenNode(
        index: m['i'] as int,
        parent: m['p'] as int,
        text: (m['text'] as String?) ?? '',
        desc: (m['desc'] as String?) ?? '',
        cls: (m['cls'] as String?) ?? '',
        clickable: m['click'] == true,
        editable: m['edit'] == true,
        scrollable: m['scroll'] == true,
        checkable: m['check'] == true,
        checked: m['checked'] == true,
        password: m['password'] == true,
        enabled: m['enabled'] != false,
        moreDown: m['more_down'] == true,
        moreUp: m['more_up'] == true,
      );

  final int index;
  final int parent;
  final String text, desc, cls;
  final bool clickable, editable, scrollable, checkable, checked, password, enabled;
  final bool moreDown, moreUp; // scrollable only: content past the screen edge

  String get label {
    final t = text.trim().isNotEmpty ? text : desc;
    return t.replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  bool get interactive => clickable || editable || scrollable || checkable;
}

enum ElementKind { button, toggle, input, list, text }

/// An element the model can see; [id] is the snapshot index the service acts on.
class ScreenElement {
  ScreenElement(this.id, this.kind, this.label, {this.on, this.moreDown = false, this.moreUp = false});
  final int? id; // null for plain text
  final ElementKind kind;
  final String label;
  final bool? on; // toggles only
  final bool moreDown, moreUp; // lists only
}

/// The compact text the model reads, and what its ids refer to.
class CompactScreen {
  CompactScreen(this.package, this.elements, this.text, {this.blocked = false, this.more = 0});
  final String package;
  final List<ScreenElement> elements;
  final String text;
  final bool blocked;
  final int more;

  ScreenElement? byId(int id) {
    for (final e in elements) {
      if (e.id == id) return e;
    }
    return null;
  }

  /// The list to scroll: the first one with more content in [down]'s
  /// direction, else the first list.
  int? mainList({bool down = true}) {
    for (final e in elements) {
      if (e.kind == ElementKind.list && (down ? e.moreDown : e.moreUp)) return e.id;
    }
    for (final e in elements) {
      if (e.kind == ElementKind.list) return e.id;
    }
    return null;
  }

  /// Every number shown on this screen (for the answer-grounding check).
  Iterable<String> get numbers sync* {
    for (final e in elements) {
      for (final m in RegExp(r'\d+(?:[.,]\d+)*').allMatches(e.label)) {
        yield m[0]!;
      }
    }
  }
}

String _clip(String s, int n) => s.length <= n ? s : '${s.substring(0, n - 1)}…';

/// Turn a raw snapshot into a short, numbered element list.
///
/// A clickable row usually has no text of its own; its title and summary sit
/// in child text nodes, so the row takes them as its label ("Wi-Fi · Home
/// network") and they are not listed again. A row holding a switch is a
/// toggle with its on/off state. Text that belongs to no interactive element
/// is listed without an id. Password fields arrive empty from the service.
///
/// With [ids] false no element shows an id: the model names what it taps
/// (DeviceTools.byLabel).
CompactScreen compactScreen(Map<Object?, Object?> snapshot,
    {int maxElements = 60, int maxLabel = 48, bool ids = true}) {
  final pkg = (snapshot['package'] as String?) ?? '';
  if (snapshot['blocked'] == true) {
    return CompactScreen(pkg, const [], 'app: $pkg\n(not an allowed app: its screen is not read)',
        blocked: true);
  }
  final nodes = [
    for (final m in (snapshot['nodes'] as List? ?? const []))
      ScreenNode.fromMap(m as Map<Object?, Object?>)
  ];
  final byIndex = {for (final n in nodes) n.index: n};
  final children = <int, List<ScreenNode>>{};
  for (final n in nodes) {
    children.putIfAbsent(n.parent, () => []).add(n);
  }

  // The nearest interactive ancestor that will be shown with an id.
  int? owner(ScreenNode n) {
    var p = byIndex[n.parent];
    while (p != null) {
      if (p.clickable || p.editable || p.checkable) return p.index;
      p = byIndex[p.parent];
    }
    return null;
  }

  // Descendant labels and switches of each interactive node, in tree order.
  final consumed = <int>{};
  final out = <ScreenElement>[];
  final seenText = <String>{};
  for (final n in nodes) {
    if (consumed.contains(n.index)) continue;
    if (n.clickable || n.editable || n.checkable) {
      final parts = <String>[];
      if (n.label.isNotEmpty) parts.add(n.label);
      bool? on = n.checkable ? n.checked : null;
      void visit(ScreenNode c) {
        for (final k in children[c.index] ?? const <ScreenNode>[]) {
          if (k.clickable || k.editable) continue; // listed on its own
          consumed.add(k.index);
          if (k.checkable) on ??= k.checked;
          if (k.label.isNotEmpty && !parts.contains(k.label) && parts.length < 3) parts.add(k.label);
          visit(k);
        }
      }

      visit(n);
      if (parts.isEmpty && on == null && !n.editable) continue; // unlabeled container
      final kind = n.editable
          ? ElementKind.input
          : on != null
              ? ElementKind.toggle
              : ElementKind.button;
      out.add(ScreenElement(n.index, kind, _clip(parts.join(' · '), maxLabel), on: on));
      seenText.addAll(parts);
    } else if (n.scrollable) {
      out.add(ScreenElement(n.index, ElementKind.list, n.label.isEmpty ? '' : _clip(n.label, maxLabel),
          moreDown: n.moreDown, moreUp: n.moreUp));
    } else if (n.label.isNotEmpty && owner(n) == null && !seenText.contains(n.label)) {
      seenText.add(n.label);
      out.add(ScreenElement(null, ElementKind.text, _clip(n.label, maxLabel)));
    }
  }

  final shown = out.take(maxElements).toList();
  final more = out.length - shown.length;
  final b = StringBuffer('app: $pkg\n');
  for (final e in shown) {
    final id = e.id == null || !ids ? '' : '[${e.id}] ';
    switch (e.kind) {
      case ElementKind.text:
        b.writeln('text "${e.label}"');
      case ElementKind.list:
        final more = [if (e.moreDown) 'more below: scroll down to see it', if (e.moreUp) 'more above'];
        b.writeln('${id}list${e.label.isEmpty ? '' : ' "${e.label}"'}'
            '${more.isEmpty ? ' (scrollable)' : ' (${more.join('; ')})'}');
      case ElementKind.toggle:
        b.writeln('${id}toggle "${e.label}" ${e.on == true ? 'on' : 'off'}');
      case ElementKind.input:
        b.writeln('${id}input "${e.label}"');
      case ElementKind.button:
        b.writeln('${id}button "${e.label}"');
    }
  }
  if (more > 0) b.writeln('($more more elements: scroll to see them)');
  return CompactScreen(pkg, shown, b.toString().trimRight(), more: more);
}

/// What the agent may do. First proof: navigate and read only.
class DevicePolicy {
  const DevicePolicy({
    this.apps = const {'settings': 'com.android.settings'},
    this.related = const {
      // ColorOS splits Settings across packages: Bluetooth, Wi-Fi and
      // hotspot pages are com.oplus.wirelesssettings, Battery is
      // com.oplus.battery.
      'com.android.settings': {'com.oplus.wirelesssettings', 'com.oplus.battery'},
    },
    this.hints = const {
      'settings': 'In Settings, phone details (Android version, model, storage) are on the "About device" '
          'page: from the main Settings list, scroll down until "About device" is shown, then tap it.',
    },
    this.pages = const {
      'main': ('android.settings.SETTINGS', 'the main Settings list'),
      'about': ('android.settings.DEVICE_INFO_SETTINGS', 'Android version, model, device name, storage, RAM, processor'),
      'battery': ('android.intent.action.POWER_USAGE_SUMMARY', 'battery level, charging, power saving'),
      'display': ('android.settings.DISPLAY_SETTINGS', 'dark mode, brightness, auto screen off, refresh rate'),
      'bluetooth': ('android.settings.BLUETOOTH_SETTINGS', 'Bluetooth on/off and devices'),
      'wifi': ('android.settings.WIFI_SETTINGS', 'Wi-Fi on/off and networks'),
      'storage': ('android.settings.INTERNAL_STORAGE_SETTINGS', 'storage use by category'),
      'sound': ('android.settings.SOUND_SETTINGS', 'volume, ringtone, vibration'),
      'apps': ('android.settings.APPLICATION_SETTINGS', 'installed apps'),
      'location': ('android.settings.LOCATION_SOURCE_SETTINGS', 'location on/off'),
      'date_time': ('android.settings.DATE_SETTINGS', 'date, time, time zone'),
      'language': ('android.settings.LOCALE_SETTINGS', 'languages'),
    },
    this.labelHints = const {
      'settings': 'In Settings, phone details (Android version, model, storage, RAM) are on the "About device" '
          'page, battery is on "Battery", dark mode and screen timeout are on "Display & brightness".',
    },
  });

  /// App name the model may use -> package.
  final Map<String, String> apps;

  /// Packages an allowed app opens for its own pages, readable as part of it.
  final Map<String, Set<String>> related;

  /// Short per-app guidance for the model.
  final Map<String, String> hints;

  /// Settings pages the agent may open directly (DeviceTools.byPage):
  /// name -> (intent action, what the page shows). SipAccessibilityService
  /// keeps its own copy of the actions and opens nothing else.
  final Map<String, (String, String)> pages;

  /// The same guidance when the model taps by name (DeviceTools.byLabel).
  final Map<String, String> labelHints;

  Set<String> get packages => {for (final p in apps.values) ...{p, ...?related[p]}};

  String get hintText => hints.values.join(' ');

  String get labelHintText => labelHints.values.join(' ');

  // Labels whose tap would change or destroy something, not just navigate.
  static final _consequential = RegExp(
      r'\b(erase|reset|delete|remove|uninstall|forget|format|factory|clear|wipe|power off|shut ?down|restart|reboot|sign out|log ?out|disable|turn off|deactivate|pay|buy|purchase|send|install|update now|allow|confirm)\b',
      caseSensitive: false);

  /// null if allowed, else the reason the model is told.
  String? checkTap(ScreenElement? e, int id) {
    if (e == null || e.id == null) return 'no element [$id] on the current screen';
    if (e.kind == ElementKind.toggle) {
      return 'not allowed: [$id] is a toggle, and changing settings needs your confirmation (not available yet)';
    }
    if (e.kind == ElementKind.input) return 'not allowed: typing is not available yet';
    if (_consequential.hasMatch(e.label)) {
      return 'not allowed: "${e.label}" could change or delete something';
    }
    return null;
  }

  String? packageFor(String name) {
    final key = name.trim().toLowerCase();
    return apps[key] ?? (apps.containsValue(name.trim()) ? name.trim() : null);
  }
}

/// Method channel to SipAccessibilityService (via MainActivity).
class DeviceBridge {
  DeviceBridge({void Function()? onStop}) {
    _channel.setMethodCallHandler((call) async {
      if (call.method == 'stop') onStop?.call();
    });
  }

  static const _channel = MethodChannel('sipllm/device');

  Future<bool> isEnabled() async => (await _channel.invokeMethod<bool>('isEnabled')) ?? false;
  Future<void> openSettings() => _channel.invokeMethod('openSettings');
  Future<bool> prepareKeepAlive() async => (await _channel.invokeMethod<bool>('prepareKeepAlive')) ?? false;
  Future<void> openAppSettings() => _channel.invokeMethod('openAppSettings');
  Future<bool> beginRun(String status, Set<String> allowed) async =>
      (await _channel.invokeMethod<bool>('beginRun', {'status': status, 'allowed': allowed.toList()})) ?? false;
  Future<void> setStatus(String status) => _channel.invokeMethod('setStatus', {'status': status});
  Future<void> endRun() => _channel.invokeMethod('endRun');
  Future<void> returnToApp() => _channel.invokeMethod('returnToApp');

  /// [returnToApp], then the package in front afterwards (this app's package
  /// when it worked; MainActivity retries once if it did not).
  Future<String?> returnToAppChecked() => _channel.invokeMethod<String>('returnToApp');
  Future<Map<Object?, Object?>> snapshot({bool settle = true}) async =>
      (await _channel.invokeMethod<Map<Object?, Object?>>('snapshot', {'settle': settle}))!;
  Future<String> act(String action, {int index = -1, String arg = ''}) async =>
      (await _channel.invokeMethod<String>('act', {'action': action, 'index': index, 'arg': arg})) ?? 'error: no reply';
}

/// The agent's device tools, and their executor.
class DeviceTools {
  DeviceTools(this.bridge, this.policy, {this.byLabel = false, this.byPage = false});
  final DeviceBridge bridge;
  final DevicePolicy policy;

  /// Experimental (default off): screens show names, not ids, and `tap` takes
  /// a name; the executor scrolls the current list until that name is on
  /// screen. Moves scrolling out of the model's hands for small models.
  final bool byLabel;

  /// Experimental (default off): an open_settings_page tool opens a Settings
  /// page directly by name (an intent), so reaching it is one choice instead
  /// of a scroll-and-tap search.
  final bool byPage;
  CompactScreen? _screen;
  CompactScreen? get screen => _screen;

  // Every number seen on any screen this run; see [ungroundedNumbers].
  final Set<String> _seenNumbers = {};

  /// Numbers in [answer] that appeared on no screen read during the run. A
  /// non-empty result means the model stated a figure it never saw.
  Set<String> ungroundedNumbers(String answer) => {
        for (final m in RegExp(r'\d+(?:[.,]\d+)*').allMatches(answer))
          if (!_seenNumbers.contains(m[0]!)) m[0]!
      };

  void resetRun() {
    _seenNumbers.clear();
    _lastAction = null;
    _repeats = 0;
    _depth = 0;
  }

  // Pages opened below the app's main screen. Back at depth 0 would leave the
  // app (and reopening it loses the scroll position), so it is refused.
  int _depth = 0;

  // Repetition guard: the same call landing on the same screen again.
  String? _lastAction;
  int _repeats = 0;

  List<AgentTool> get tools => byPage
      ? [
          AgentTool('open_settings_page',
              'Open a Settings page directly. Pages: ${[for (final e in policy.pages.entries) '${e.key} (${e.value.$2})'].join('; ')}.',
              const [AgentToolParam('page', description: 'one page name from the list, e.g. "about"')]),
          const AgentTool('tap', 'Tap the element with this id on the current screen.',
              [AgentToolParam('id', type: AgentParamType.integer, description: 'element id from the screen')]),
          const AgentTool('scroll', 'Scroll the screen to see more of this page.',
              [AgentToolParam('direction', description: '"down" or "up"')]),
          const AgentTool('back', 'Press the Back button.'),
        ]
      : byLabel
      ? [
          const AgentTool('tap',
              'Tap the item with this name. If it is not on screen, the list is scrolled until it is.',
              [AgentToolParam('name', description: 'the item name as the screen shows it, e.g. "About device"')]),
          const AgentTool('scroll', 'Scroll the screen to see more of this page.',
              [AgentToolParam('direction', description: '"down" or "up"')]),
          const AgentTool('back', 'Go back to the previous page.'),
        ]
      : [
        const AgentTool('tap', 'Tap the element with this id on the current screen.',
            [AgentToolParam('id', type: AgentParamType.integer, description: 'element id from the screen')]),
        const AgentTool('scroll', 'Scroll the screen to see more elements.',
            [AgentToolParam('direction', description: '"down" or "up"')]),
        const AgentTool('back', 'Press the Back button.'),
        AgentTool('open_app', 'Open an app. Allowed: ${policy.apps.keys.join(', ')}.',
            const [AgentToolParam('name', description: 'app name')]),
      ];

  /// Read the screen now (after the UI settles).
  /// When set, each raw snapshot is passed here (the app saves the last one
  /// next to the run transcript, to debug what the compactor dropped).
  void Function(Map<Object?, Object?> raw)? onRawSnapshot;

  Future<CompactScreen> read() async {
    final raw = await bridge.snapshot();
    onRawSnapshot?.call(raw);
    _screen = compactScreen(raw, ids: !byLabel);
    _seenNumbers.addAll(_screen!.numbers);
    return _screen!;
  }

  Future<AgentToolResult> _after(String outcome) async {
    var s = await read();
    var ok = !outcome.startsWith('error');
    var head = ok ? 'done.' : outcome;
    if (s.blocked) {
      // The action left the allowed apps (Back from a top screen, a link to
      // another app). Go back to the app the run is about, and say so.
      final home = policy.apps.values.first;
      await bridge.act('launch', arg: home);
      _depth = 0;
      s = await read();
      ok = false;
      head = 'That left the allowed apps (${s.package.isEmpty ? 'another app' : 'it opened another app'}), '
          'so ${policy.apps.keys.first} was reopened at its main screen.';
    }
    return AgentToolResult('$head\nScreen now:\n${s.text}', ok: ok);
  }

  Future<AgentToolResult> execute(AgentToolCall call) async {
    // A small model can repeat one action forever (Back on a top screen was
    // pressed 10 times in one run). Twice with nothing new is enough.
    final key = '${call.name} ${call.arguments}';
    final before = _screen?.text;
    if (key == _lastAction && _repeats >= 2) {
      return AgentToolResult.error('error: "${call.name}" was already done twice and the screen did not change. '
          'Do something different: tap an item on the screen, or scroll down.\nScreen now:\n${before ?? ''}');
    }
    final r = await _execute(call);
    final same = _screen?.text == before;
    _repeats = (key == _lastAction && same) ? _repeats + 1 : (same ? 1 : 0);
    _lastAction = key;
    return r;
  }

  Future<AgentToolResult> _execute(AgentToolCall call) async {
    switch (call.name) {
      case 'back':
        if (_depth <= 0) {
          return AgentToolResult.error('error: this is the main ${policy.apps.keys.first} screen, so Back would leave it. '
              'Scroll down to see more, or tap an item.\nScreen now:\n${_screen?.text ?? ''}');
        }
        final r = await _after(await bridge.act('back'));
        if (r.ok) _depth--;
        return r;
      case 'open_app':
        final pkg = policy.packageFor(call.arguments['name'] ?? '');
        if (pkg == null) {
          return AgentToolResult.error('error: not allowed to open "${call.arguments['name']}". '
              'Allowed: ${policy.apps.keys.join(', ')}');
        }
        _depth = 0;
        return _after(await bridge.act('launch', arg: pkg));
      case 'open_settings_page' when byPage:
        final page = (call.arguments['page'] ?? '').trim().toLowerCase().replaceAll(RegExp(r'[\s-]+'), '_');
        if (!policy.pages.containsKey(page)) {
          return AgentToolResult.error('error: no page "$page". Pages: ${policy.pages.keys.join(', ')}');
        }
        _depth = page == 'main' ? 0 : 1;
        return _after(await bridge.act('open_page', arg: page));
      case 'tap' when byLabel:
        return _tapByName((call.arguments['name'] ?? call.arguments['id'] ?? '').trim());
      case 'tap':
        final id = int.tryParse((call.arguments['id'] ?? '').replaceAll(RegExp(r'[^0-9-]'), ''));
        if (id == null) return const AgentToolResult.error('error: tap needs a numeric id');
        final why = policy.checkTap(_screen?.byId(id), id);
        if (why != null) return AgentToolResult.error('error: $why');
        final before = _screen?.text;
        final r = await _after(await bridge.act('tap', index: id));
        if (r.ok && _screen?.text != before) _depth++;  // a new page opened
        return r;
      case 'scroll':
        final dir = (call.arguments['direction'] ?? 'down').toLowerCase();
        final list = _screen?.mainList(down: !dir.startsWith('u'));
        if (list == null) return _after('error: nothing on this screen scrolls');
        return _after(await bridge.act(dir.startsWith('u') ? 'scroll_backward' : 'scroll_forward', index: list));
    }
    return AgentToolResult.error('error: unknown tool ${call.name}');
  }

  // The element on the current screen called [name]: its title (the label
  // before " · ") first, then any label containing it.
  ScreenElement? _named(String name) {
    final want = name.toLowerCase().replaceAll(RegExp(r'^\[\d+\]\s*'), '').replaceAll('"', '').trim();
    if (want.isEmpty) return null;
    final els = [for (final e in _screen?.elements ?? const <ScreenElement>[]) if (e.id != null && e.kind != ElementKind.list) e];
    String title(ScreenElement e) => e.label.split(' · ').first.toLowerCase();
    for (final e in els) {
      if (title(e) == want) return e;
    }
    for (final e in els) {
      if (e.label.toLowerCase().contains(want)) return e;
    }
    return null;
  }

  Future<AgentToolResult> _tapByName(String name) async {
    if (name.isEmpty) return const AgentToolResult.error('error: tap needs the name of an item on the screen');
    var e = _named(name);
    // Not on screen: scroll down to the end of the list, then up to its top.
    for (final down in [true, false]) {
      for (var i = 0; e == null && i < 6; i++) {
        final list = _screen?.mainList(down: down);
        if (list == null) break;
        final before = _screen?.text;
        final r = await bridge.act(down ? 'scroll_forward' : 'scroll_backward', index: list);
        await read();
        if (r.startsWith('error') || _screen?.text == before) break;
        e = _named(name);
      }
      if (e != null) break;
    }
    if (e == null) {
      return AgentToolResult.error('error: no item named "$name" on this page.\nScreen now:\n${_screen?.text ?? ''}');
    }
    final why = policy.checkTap(e, e.id!);
    if (why != null) return AgentToolResult.error('error: ${why.replaceAll('[${e.id}]', '"${e.label}"')}');
    final before = _screen?.text;
    final r = await _after(await bridge.act('tap', index: e.id!));
    if (r.ok && _screen?.text != before) _depth++;
    return r;
  }
}
