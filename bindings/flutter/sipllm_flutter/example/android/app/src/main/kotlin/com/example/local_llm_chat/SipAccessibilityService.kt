package com.example.local_llm_chat

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Rect
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.atomic.AtomicReference

/**
 * The agent's hands on the device, kept thin: a snapshot of the active
 * window's visible nodes, and click / scroll / back / home / launch by the
 * snapshot's node index. Which apps and actions are allowed is decided in Dart
 * (device_policy.dart); this class enforces only what must hold even if the
 * Dart side is wrong:
 *  - nothing acts, and no snapshot is taken, unless an agent run is active;
 *  - only the run's allowlisted apps are read, acted on or launched: for any
 *    other app the snapshot carries its package name and no node text;
 *  - password fields are never read;
 *  - this app and the system UI are never read or acted on, allowlisted or not.
 * Outside a run, accessibility events are dropped without being looked at.
 */
class SipAccessibilityService : AccessibilityService() {

    // A node from the last snapshot plus what is needed to act on it again.
    private class Snap(val node: AccessibilityNodeInfo)

    private val main = Handler(Looper.getMainLooper())
    @Volatile private var lastEventMs = 0L
    @Volatile private var snaps: List<Snap> = emptyList()
    private var overlay: LinearLayout? = null
    private var overlayText: TextView? = null

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance.set(this)
    }

    override fun onDestroy() {
        instance.set(null)
        removeOverlay()
        super.onDestroy()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (!runActive) return
        lastEventMs = SystemClock.uptimeMillis()
    }

    override fun onInterrupt() {}

    // ---- snapshot ------------------------------------------------------------

    private fun snapshot(): Map<String, Any?> {
        val root = rootInActiveWindow ?: return mapOf("package" to "", "nodes" to emptyList<Any>())
        val pkg = root.packageName?.toString() ?: ""
        if (!readable(pkg)) {
            snaps = emptyList()
            return mapOf("package" to pkg, "nodes" to emptyList<Any>(), "blocked" to true)
        }
        val list = ArrayList<Snap>()
        val out = ArrayList<Map<String, Any?>>()
        fun walk(n: AccessibilityNodeInfo?, parent: Int, depth: Int) {
            if (n == null || depth > 40 || out.size >= 400) return
            if (!n.isVisibleToUser) return
            val idx = out.size
            val r = Rect()
            n.getBoundsInScreen(r)
            val password = n.isPassword
            list.add(Snap(n))
            out.add(mapOf(
                "i" to idx,
                "p" to parent,
                "text" to if (password) "" else (n.text?.toString() ?: ""),
                "desc" to if (password) "" else (n.contentDescription?.toString() ?: ""),
                "cls" to (n.className?.toString()?.substringAfterLast('.') ?: ""),
                "click" to n.isClickable,
                "edit" to n.isEditable,
                "scroll" to n.isScrollable,
                // Whether the list has more content past what is on screen; the
                // model will not scroll unless it is told something is there.
                "more_down" to n.actionList.contains(AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_FORWARD),
                "more_up" to n.actionList.contains(AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_BACKWARD),
                "check" to n.isCheckable,
                "checked" to n.isChecked,
                "password" to password,
                "enabled" to n.isEnabled,
                "b" to listOf(r.left, r.top, r.right, r.bottom),
            ))
            for (c in 0 until n.childCount) walk(n.getChild(c), idx, depth + 1)
        }
        walk(root, -1, 0)
        snaps = list
        return mapOf("package" to pkg, "nodes" to out)
    }

    // ---- actions ---------------------------------------------------------------

    private fun readable(pkg: String): Boolean =
        pkg != packageName && pkg != "com.android.systemui" && pkg in allowed

    private fun blockedPackage(): String? {
        val pkg = rootInActiveWindow?.packageName?.toString() ?: return "unknown"
        return if (readable(pkg)) null else pkg
    }

    /** Returns "ok" or an error string the model can read. */
    private fun act(action: String, index: Int, arg: String): String {
        when (action) {
            "back" -> return if (performGlobalAction(GLOBAL_ACTION_BACK)) "ok" else "error: back failed"
            "home" -> return if (performGlobalAction(GLOBAL_ACTION_HOME)) "ok" else "error: home failed"
            "launch" -> {
                if (!readable(arg)) return "error: not allowed to open $arg"
                val intent = packageManager.getLaunchIntentForPackage(arg)
                    ?: return "error: app not installed: $arg"
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                startActivity(intent)
                return "ok"
            }
            // A Settings page by name. Only the actions in PAGES, and only when
            // the activity that handles one belongs to an allowed package.
            "open_page" -> {
                val action = PAGES[arg] ?: return "error: no page $arg"
                val intent = Intent(action)
                val target = intent.resolveActivity(packageManager)?.packageName
                    ?: return "error: this phone has no $arg page"
                if (!readable(target)) return "error: the $arg page is in $target, which is not allowed"
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                startActivity(intent)
                return "ok"
            }
        }
        blockedPackage()?.let { return "error: not allowed to act in $it" }
        val snap = snaps.getOrNull(index) ?: return "error: no element [$index] on the last screen"
        // A stale snapshot must never act on whatever now sits in its place.
        if (!snap.node.refresh() || !snap.node.isVisibleToUser) return "error: screen changed, element [$index] is gone"
        return when (action) {
            "tap" -> {
                var n: AccessibilityNodeInfo? = snap.node
                while (n != null && !n.isClickable) n = n.parent
                when {
                    n == null -> "error: element [$index] is not tappable"
                    !n.isEnabled -> "error: element [$index] is disabled"
                    n.performAction(AccessibilityNodeInfo.ACTION_CLICK) -> "ok"
                    else -> "error: tap on [$index] was refused"
                }
            }
            "scroll_forward", "scroll_backward" -> {
                var n: AccessibilityNodeInfo? = snap.node
                while (n != null && !n.isScrollable) n = n.parent
                val a = if (action == "scroll_forward") AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
                        else AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
                when {
                    n == null -> "error: element [$index] is not scrollable"
                    n.performAction(a) -> "ok"
                    else -> "error: cannot scroll further"
                }
            }
            else -> "error: unknown action $action"
        }
    }

    /** Wait until no window change for [quietMs], at most [maxMs]. Off the main thread. */
    private fun settle(quietMs: Long = 400, maxMs: Long = 3000) {
        val start = SystemClock.uptimeMillis()
        lastEventMs = start
        while (true) {
            val now = SystemClock.uptimeMillis()
            if (now - lastEventMs >= quietMs || now - start >= maxMs) return
            Thread.sleep(50)
        }
    }

    // ---- overlay: status + Stop, visible over any app --------------------------

    private fun showOverlay(status: String) {
        if (overlay != null) { overlayText?.text = status; return }
        val wm = getSystemService(WINDOW_SERVICE) as WindowManager
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.argb(230, 20, 22, 28))
            setPadding(24, 12, 12, 12)
        }
        val label = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 13f
            text = status
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        val stop = Button(this).apply {
            text = "Stop"
            setOnClickListener { stopListener?.invoke() }
        }
        box.addView(label)
        box.addView(stop)
        val lp = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT,
        ).apply { gravity = Gravity.BOTTOM }
        wm.addView(box, lp)
        overlay = box
        overlayText = label
    }

    private fun removeOverlay() {
        val v = overlay ?: return
        try { (getSystemService(WINDOW_SERVICE) as WindowManager).removeView(v) } catch (_: Exception) {}
        overlay = null
        overlayText = null
    }

    companion object {
        // Settings pages open_page may open (device_control.dart DevicePolicy.pages).
        private val PAGES = mapOf(
            "main" to "android.settings.SETTINGS",
            "about" to "android.settings.DEVICE_INFO_SETTINGS",
            "battery" to "android.intent.action.POWER_USAGE_SUMMARY",
            "display" to "android.settings.DISPLAY_SETTINGS",
            "bluetooth" to "android.settings.BLUETOOTH_SETTINGS",
            "wifi" to "android.settings.WIFI_SETTINGS",
            "storage" to "android.settings.INTERNAL_STORAGE_SETTINGS",
            "sound" to "android.settings.SOUND_SETTINGS",
            "apps" to "android.settings.APPLICATION_SETTINGS",
            "location" to "android.settings.LOCATION_SOURCE_SETTINGS",
            "date_time" to "android.settings.DATE_SETTINGS",
            "language" to "android.settings.LOCALE_SETTINGS",
        )
        private val instance = AtomicReference<SipAccessibilityService?>()
        @Volatile private var runActive = false
        @Volatile private var allowed: Set<String> = emptySet()
        @Volatile var stopListener: (() -> Unit)? = null

        fun isEnabled(): Boolean = instance.get() != null

        /** Main thread. */
        fun beginRun(status: String, allowedPackages: Set<String>): Boolean {
            val s = instance.get() ?: return false
            allowed = allowedPackages
            runActive = true
            s.showOverlay(status)
            return true
        }

        /** Main thread. */
        fun setStatus(status: String) { instance.get()?.overlayText?.text = status }

        /** Main thread. */
        fun endRun() {
            runActive = false
            allowed = emptySet()
            instance.get()?.let { it.snaps = emptyList(); it.removeOverlay() }
        }

        /** Any thread except main (it may wait for the UI to settle). */
        fun snapshot(settleFirst: Boolean): Map<String, Any?> {
            val s = instance.get() ?: throw IllegalStateException("accessibility service is not enabled")
            check(runActive) { "no agent run is active" }
            if (settleFirst) s.settle()
            return s.snapshot()
        }

        /** Any thread except main. Returns "ok" or an error message. */
        fun act(action: String, index: Int, arg: String): String {
            val s = instance.get() ?: return "error: accessibility service is not enabled"
            if (!runActive) return "error: no agent run is active"
            return s.act(action, index, arg)
        }

        fun currentPackage(): String = instance.get()?.rootInActiveWindow?.packageName?.toString() ?: ""

        /** Main thread. Brings this app's activity to the front, started from the
         *  service; false if the service is not running or the start threw. */
        fun bringAppToFront(): Boolean {
            val s = instance.get() ?: return false
            return try {
                s.startActivity(Intent(s, MainActivity::class.java)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT))
                true
            } catch (_: Exception) {
                false
            }
        }
    }
}
