// device_control_test.dart — the screen compactor and the device policy, on a
// real Settings home-screen tree (OnePlus IV2201, captured with uiautomator and
// converted to SipAccessibilityService's snapshot format; the Wi-Fi network
// name was replaced).
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:local_llm_chat/core/device/device_control.dart';
import 'package:sipllm_flutter/sipllm_flutter.dart';

// A bridge whose screen never changes and that records every action.
class _FakeBridge extends DeviceBridge {
  final acts = <String>[];
  @override
  Future<Map<Object?, Object?>> snapshot({bool settle = true}) async => {
        'package': 'com.android.settings',
        'nodes': [
          {'i': 0, 'p': -1, 'text': 'Display', 'click': true},
          {'i': 1, 'p': -1, 'scroll': true, 'more_down': true},
        ],
      };
  @override
  Future<String> act(String action, {int index = -1, String arg = ''}) async {
    acts.add(action);
    return 'ok';
  }
}

Map<Object?, Object?> _fixture() =>
    jsonDecode(File('test/fixtures/settings_home.json').readAsStringSync()) as Map<Object?, Object?>;

Map<String, Object?> _n(int i, int p,
        {String text = '', bool click = false, bool check = false, bool checked = false,
        bool edit = false, bool password = false, bool scroll = false}) =>
    {'i': i, 'p': p, 'text': text, 'desc': '', 'cls': '', 'click': click, 'edit': edit,
     'scroll': scroll, 'check': check, 'checked': checked, 'password': password, 'enabled': true};

void main() {
  test('Settings home compacts to labelled rows with ids', () {
    final s = compactScreen(_fixture());
    expect(s.package, 'com.android.settings');
    final lines = s.text.split('\n');
    expect(lines.first, 'app: com.android.settings');
    // Rows take their child titles and summaries as one label.
    expect(lines, contains('[27] button "Wi-Fi · ExampleNet"'));
    expect(lines, contains('[36] button "Bluetooth · Off"'));
    expect(lines, contains('[54] button "Display & brightness"'));
    expect(lines, contains('[62] button "Search"'));
    expect(lines, contains('[9] list (scrollable)'));   // uiautomator dumps carry no scroll actions
    // Child texts are not listed again, duplicates collapse, and unlabeled
    // containers are dropped.
    expect(lines.where((l) => l.contains('"Wi-Fi')).length, 1);
    expect(lines.where((l) => l.contains('Settings')).length, 1);
    expect(lines.any((l) => l.startsWith('[10] ')), isFalse);
    // Small enough to prefill every step: ~3.5 characters per Qwen token.
    expect(s.text.length, lessThan(900), reason: s.text);
    expect(s.mainList(), 9);
  });

  test('a row with a switch is a toggle with its state; passwords stay empty', () {
    final s = compactScreen({
      'package': 'com.android.settings',
      'nodes': [
        _n(0, -1, scroll: true),
        _n(1, 0, click: true),
        _n(2, 1, text: 'Dark mode'),
        _n(3, 1, check: true, checked: true),
        _n(4, 0, edit: true, password: true),
        _n(5, -1, text: 'Android version 15'),
      ],
    });
    expect(s.text.split('\n'), containsAll(['[1] toggle "Dark mode" on', '[4] input ""', 'text "Android version 15"']));
    expect(s.byId(1)!.kind, ElementKind.toggle);
  });

  test('a list with content past the screen says so', () {
    final s = compactScreen({
      'package': 'p',
      'nodes': [
        {..._n(0, -1, scroll: true), 'more_down': true},
        _n(1, 0, click: true, text: 'About device'),
      ],
    });
    expect(s.text.split('\n'), contains('[0] list (more below: scroll down to see it)'));
    expect(s.mainList(down: true), 0);
    expect(s.numbers, isEmpty);
  });

  test('numbers in an answer are checked against what was on screen', () {
    final s = compactScreen({'package': 'p', 'nodes': [_n(0, -1, text: 'Android version 13')]});
    expect(s.numbers.toList(), ['13']);
  });

  test('a blocked app is reported by package only', () {
    final s = compactScreen({'package': 'com.example.work', 'nodes': const [], 'blocked': true});
    expect(s.blocked, isTrue);
    expect(s.text, contains('not an allowed app'));
    expect(s.elements, isEmpty);
  });

  test('the element cap keeps the prompt bounded', () {
    final nodes = [_n(0, -1, scroll: true)];
    for (var i = 1; i <= 100; i++) {
      nodes.add(_n(i, 0, click: true, text: 'Item $i'));
    }
    final s = compactScreen({'package': 'p', 'nodes': nodes}, maxElements: 20);
    expect(s.elements.length, 20);
    expect(s.text, contains('(81 more elements: scroll to see them)'));
  });

  test('policy: navigation taps only', () {
    const policy = DevicePolicy();
    final s = compactScreen({
      'package': 'com.android.settings',
      'nodes': [
        _n(0, -1, click: true, text: 'Display & brightness'),
        _n(1, -1, click: true),
        _n(2, 1, text: 'Wi-Fi'),
        _n(3, 1, check: true),
        _n(4, -1, click: true, text: 'Erase all data (factory reset)'),
        _n(5, -1, edit: true, text: 'Search settings'),
        _n(6, -1, text: 'About device'),
      ],
    });
    expect(policy.checkTap(s.byId(0), 0), isNull);
    expect(policy.checkTap(s.byId(1), 1), contains('toggle'));
    expect(policy.checkTap(s.byId(4), 4), contains('could change or delete'));
    expect(policy.checkTap(s.byId(5), 5), contains('typing'));
    expect(policy.checkTap(s.byId(99), 99), contains('no element [99]'));
    expect(policy.checkTap(null, 6), contains('no element [6]'));
    expect(policy.packageFor('Settings'), 'com.android.settings');
    expect(policy.packageFor('WhatsApp'), isNull);
    // Settings' own pages in other ColorOS packages are readable, nothing else.
    expect(policy.packages, {'com.android.settings', 'com.oplus.wirelesssettings', 'com.oplus.battery'});
    expect(policy.hintText, contains('scroll down until "About device"'));
  });

  test('the same no-op action is refused on its third try', () async {
    TestWidgetsFlutterBinding.ensureInitialized();
    final bridge = _FakeBridge();
    final tools = DeviceTools(bridge, const DevicePolicy());
    await tools.read();
    const scroll = AgentToolCall(0, 'scroll', {'direction': 'down'}, '{}');
    await tools.execute(scroll);
    await tools.execute(scroll);
    final third = await tools.execute(scroll);
    expect(third.ok, isFalse);
    expect(third.text, contains('already done twice'));
    expect(bridge.acts, ['scroll_forward', 'scroll_forward']); // the third never reached the phone
    // A different action is allowed again.
    expect((await tools.execute(const AgentToolCall(0, 'tap', {'id': '0'}, '{}'))).ok, isTrue);
  });

  test('back on the main screen is refused without touching the phone', () async {
    TestWidgetsFlutterBinding.ensureInitialized();
    final bridge = _FakeBridge();
    final tools = DeviceTools(bridge, const DevicePolicy())..resetRun();
    await tools.read();
    final r = await tools.execute(const AgentToolCall(0, 'back', {}, '{}'));
    expect(r.ok, isFalse);
    expect(r.text, contains('Back would leave it'));
    expect(bridge.acts, isEmpty);
  });
}
