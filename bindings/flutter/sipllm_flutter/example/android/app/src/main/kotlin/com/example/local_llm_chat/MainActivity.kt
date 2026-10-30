package com.example.local_llm_chat

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.PowerManager
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.util.concurrent.Executors

class MainActivity : FlutterActivity() {
    private val main = Handler(Looper.getMainLooper())
    // Snapshots and actions wait for the UI to settle: never on the main thread.
    private val worker = Executors.newSingleThreadExecutor()
    // Whether this activity is in front; read off the main thread by returnToApp.
    @Volatile private var resumed = false

    override fun onResume() {
        super.onResume()
        resumed = true
    }

    override fun onPause() {
        resumed = false
        super.onPause()
    }

    private fun bringToFront() {
        startActivity(Intent(this, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT or Intent.FLAG_ACTIVITY_NEW_TASK))
    }

    /** Off the main thread: waits up to [ms] for this app to be in front, then
     *  returns the package in front (this app's when it is). */
    private fun frontAfter(ms: Long): String {
        val end = SystemClock.uptimeMillis() + ms
        while (true) {
            if (resumed) return packageName
            if (SystemClock.uptimeMillis() >= end) {
                return SipAccessibilityService.currentPackage().ifEmpty { "unknown" }
            }
            Thread.sleep(100)
        }
    }

    /**
     * Back on the root screen moves the app to the background instead of
     * finishing the activity. Finishing destroyed the Flutter engine while the
     * process lived on, so a loaded model (~1.8 GB of native memory) stayed
     * allocated with nothing able to free it, and an agent run driving another
     * app died with the engine.
     */
    override fun popSystemNavigator(): Boolean {
        moveTaskToBack(true)
        return true
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val channel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "sipllm/device")
        SipAccessibilityService.stopListener = { main.post { channel.invokeMethod("stop", null) } }
        channel.setMethodCallHandler { call, result ->
            when (call.method) {
                "isEnabled" -> result.success(SipAccessibilityService.isEnabled())
                "openSettings" -> {
                    startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                    result.success(null)
                }
                // Called when the user picks Phone mode: ask once for what keeps a
                // run alive in the background. Each is a no-op once granted.
                "prepareKeepAlive" -> {
                    if (Build.VERSION.SDK_INT >= 33 &&
                        checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                        requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 7302)
                    }
                    val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
                    val exempt = pm.isIgnoringBatteryOptimizations(packageName)
                    if (!exempt) {
                        try {
                            startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                                Uri.parse("package:$packageName")))
                        } catch (_: Exception) {}
                    }
                    result.success(exempt)
                }
                // The app's own settings page: on ColorOS, Battery usage has
                // "Allow background activity", which its CPU killer honours.
                "openAppSettings" -> {
                    startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                        Uri.parse("package:$packageName")).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                    result.success(null)
                }
                "beginRun" -> {
                    val allowed = (call.argument<List<String>>("allowed") ?: emptyList()).toSet()
                    val status = call.argument<String>("status") ?: ""
                    val ok = SipAccessibilityService.beginRun(status, allowed)
                    // Started while this activity is in front, as Android requires.
                    if (ok) AgentRunService.start(this, status)
                    result.success(ok)
                }
                "setStatus" -> {
                    SipAccessibilityService.setStatus(call.argument<String>("status") ?: "")
                    result.success(null)
                }
                "endRun" -> {
                    SipAccessibilityService.endRun()
                    AgentRunService.stop(this)
                    result.success(null)
                }
                // Android can refuse an activity start from the background without
                // an error, which left Settings in front after a run. Check, retry
                // once from the accessibility service, and reply with the package
                // in front so the run's transcript records what happened.
                "returnToApp" -> {
                    try { bringToFront() } catch (_: Exception) {}
                    try {
                        worker.execute {
                            // Always reply: Dart ends the run (overlay, service) after it.
                            val reply = try {
                                var front = frontAfter(1500)
                                if (front != packageName) {
                                    main.post {
                                        if (!SipAccessibilityService.bringAppToFront()) {
                                            try { bringToFront() } catch (_: Exception) {}
                                        }
                                    }
                                    front = frontAfter(1500)
                                    front = if (front == packageName) "$front (after retry)" else front
                                }
                                front
                            } catch (e: Exception) {
                                "error: ${e.message}"
                            }
                            main.post { result.success(reply) }
                        }
                    } catch (e: Exception) {
                        result.success("unknown: ${e.message}")
                    }
                }
                "snapshot" -> {
                    val settle = call.argument<Boolean>("settle") ?: true
                    worker.execute {
                        try {
                            val snap = SipAccessibilityService.snapshot(settle)
                            main.post { result.success(snap) }
                        } catch (e: Exception) {
                            main.post { result.error("snapshot", e.message, null) }
                        }
                    }
                }
                "act" -> {
                    val action = call.argument<String>("action") ?: ""
                    val index = call.argument<Int>("index") ?: -1
                    val arg = call.argument<String>("arg") ?: ""
                    worker.execute {
                        val r = try { SipAccessibilityService.act(action, index, arg) }
                                catch (e: Exception) { "error: ${e.message}" }
                        main.post { result.success(r) }
                    }
                }
                else -> result.notImplemented()
            }
        }
    }

    override fun onDestroy() {
        SipAccessibilityService.stopListener = null
        worker.shutdown()
        super.onDestroy()
    }
}
