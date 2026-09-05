package com.whalegirl.pet;
import android.app.*;import android.content.*;import android.os.Build;
public class ReminderReceiver extends BroadcastReceiver{
 public void onReceive(Context c,Intent i){String t=i.getStringExtra("title");if(t==null)t="有一项待办到时间啦";String ch="whale_todo";NotificationManager nm=(NotificationManager)c.getSystemService(Context.NOTIFICATION_SERVICE);if(Build.VERSION.SDK_INT>=26)nm.createNotificationChannel(new NotificationChannel(ch,"鲸鱼娘待办提醒",NotificationManager.IMPORTANCE_HIGH));nm.notify((int)(System.currentTimeMillis()%100000),new Notification.Builder(c,ch).setSmallIcon(android.R.drawable.ic_dialog_info).setContentTitle("鲸鱼娘提醒你").setContentText(t).setAutoCancel(true).build());Intent s=new Intent(c,PetOverlayService.class).putExtra("speak","待办「"+t+"」到时间啦～");if(Build.VERSION.SDK_INT>=26)c.startForegroundService(s);else c.startService(s);}
}
