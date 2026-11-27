package com.example.local_llm_chat

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager

/**
 * Foreground service held for exactly the length of a Phone-mode agent run.
 *
 * While the agent drives another app, this app is in the background running
 * inference on every core with the model resident. On the IV2201, ColorOS's
 * CPU manager killed it 70 s into a run (am_kill "o-kill(46) ... with
 * cpumanager"), taking the run with it. A foreground service marks the work as
 * user-started and visible. It holds no state: MainActivity starts it in
 * beginRun and stops it in endRun.
 */
class AgentRunService : Service() {
    private var wake: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        wake?.let { if (it.isHeld) it.release() }
        wake = null
        super.onDestroy()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Agent runs", NotificationManager.IMPORTANCE_LOW))
        }
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT), PendingIntent.FLAG_IMMUTABLE)
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) Notification.Builder(this, CHANNEL)
                      else @Suppress("DEPRECATION") Notification.Builder(this)
        val n = builder
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setContentTitle("SipLLM agent is operating your phone")
            .setContentText(intent?.getStringExtra("status") ?: "Running on-device. Use the Stop bar to end it.")
            .setContentIntent(open)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(ID, n)
        }
        // Inference must not stall if the screen goes off mid-run. Bounded, in
        // case endRun never comes (the run's process was killed).
        if (wake == null) {
            wake = (getSystemService(Context.POWER_SERVICE) as PowerManager)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "sipllm:agent-run")
                .apply { acquire(30 * 60 * 1000L) }
        }
        return START_NOT_STICKY
    }

    companion object {
        private const val CHANNEL = "sipllm_agent"
        private const val ID = 7301

        fun start(ctx: Context, status: String) {
            val i = Intent(ctx, AgentRunService::class.java).putExtra("status", status)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ctx.startForegroundService(i) else ctx.startService(i)
        }

        fun stop(ctx: Context) {
            ctx.stopService(Intent(ctx, AgentRunService::class.java))
        }
    }
}
