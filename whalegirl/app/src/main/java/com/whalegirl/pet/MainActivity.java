package com.whalegirl.pet;

import android.Manifest;
import android.app.*;
import android.content.*;
import android.graphics.Color;
import android.net.Uri;
import android.os.*;
import android.provider.Settings;
import android.view.Gravity;
import android.widget.*;
import org.json.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class MainActivity extends Activity {
  private LinearLayout listBox;
  private SharedPreferences prefs;
  @Override public void onCreate(Bundle b){ super.onCreate(b); prefs=getSharedPreferences("todos",MODE_PRIVATE); if(Build.VERSION.SDK_INT>=33) requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS},9); render(); }
  private void render(){
    LinearLayout root=new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL); root.setPadding(36,36,36,36); root.setBackgroundColor(Color.rgb(245,248,255));
    TextView title=new TextView(this); title.setText("鲸鱼娘桌宠 V0.3.1"); title.setTextSize(27); title.setGravity(Gravity.CENTER); root.addView(title);
    TextView sub=new TextView(this); sub.setText("拖拽 · 重力弹跳 · 待办提醒"); sub.setGravity(Gravity.CENTER); sub.setPadding(0,10,0,18); root.addView(sub);
    LinearLayout row=new LinearLayout(this);
    Button start=new Button(this); start.setText("启动鲸鱼娘"); start.setOnClickListener(v->startPet()); row.addView(start,new LinearLayout.LayoutParams(0,-2,1));
    Button stop=new Button(this); stop.setText("收起"); stop.setOnClickListener(v->stopService(new Intent(this,PetOverlayService.class))); row.addView(stop,new LinearLayout.LayoutParams(0,-2,1)); root.addView(row);
    Button add=new Button(this); add.setText("＋ 添加待办"); add.setOnClickListener(v->askTodo()); root.addView(add);
    listBox=new LinearLayout(this); listBox.setOrientation(LinearLayout.VERTICAL); root.addView(listBox); refresh();
    ScrollView sv=new ScrollView(this); sv.addView(root); setContentView(sv);
  }
  private JSONArray load(){ try{return new JSONArray(prefs.getString("items","[]"));}catch(Exception e){return new JSONArray();} }
  private void save(JSONArray a){ prefs.edit().putString("items",a.toString()).apply(); }
  private void refresh(){ listBox.removeAllViews(); JSONArray a=load(); if(a.length()==0){TextView t=new TextView(this);t.setText("还没有待办。鲸鱼娘暂时可以安心摸鱼～");t.setPadding(8,20,8,20);listBox.addView(t);return;} SimpleDateFormat f=new SimpleDateFormat("MM-dd HH:mm",Locale.getDefault()); for(int i=0;i<a.length();i++){ try{JSONObject o=a.getJSONObject(i); LinearLayout line=new LinearLayout(this); line.setGravity(Gravity.CENTER_VERTICAL); CheckBox cb=new CheckBox(this); cb.setText(o.getString("title")+"\n"+f.format(new Date(o.getLong("due")))); cb.setChecked(o.optBoolean("done")); final int idx=i; cb.setOnCheckedChangeListener((b,v)->{try{JSONArray x=load();x.getJSONObject(idx).put("done",v);save(x);}catch(Exception ignored){}}); line.addView(cb,new LinearLayout.LayoutParams(0,-2,1)); Button del=new Button(this);del.setText("删");del.setOnClickListener(v->{JSONArray x=load();JSONArray y=new JSONArray();for(int k=0;k<x.length();k++)if(k!=idx)y.put(x.opt(k));save(y);refresh();});line.addView(del);listBox.addView(line);}catch(Exception ignored){} } }
  private void askTodo(){ EditText e=new EditText(this);e.setHint("例如：交高数作业");new AlertDialog.Builder(this).setTitle("添加待办").setView(e).setPositiveButton("下一步",(d,w)->{if(e.length()>0) pickWhen(e.getText().toString());}).setNegativeButton("取消",null).show(); }
  private void pickWhen(String title){ Calendar c=Calendar.getInstance();c.add(Calendar.HOUR_OF_DAY,1);new DatePickerDialog(this,(dp,y,m,d)->new TimePickerDialog(this,(tp,h,min)->{c.set(y,m,d,h,min,0);long id=System.currentTimeMillis();try{JSONArray a=load();a.put(new JSONObject().put("id",id).put("title",title).put("due",c.getTimeInMillis()).put("done",false));save(a);schedule(id,title,c.getTimeInMillis());refresh();}catch(Exception ignored){}},c.get(Calendar.HOUR_OF_DAY),c.get(Calendar.MINUTE),true).show(),c.get(Calendar.YEAR),c.get(Calendar.MONTH),c.get(Calendar.DAY_OF_MONTH)).show(); }
  private void schedule(long id,String title,long at){ AlarmManager am=(AlarmManager)getSystemService(ALARM_SERVICE); Intent i=new Intent(this,ReminderReceiver.class).putExtra("title",title); PendingIntent pi=PendingIntent.getBroadcast(this,(int)(id%Integer.MAX_VALUE),i,PendingIntent.FLAG_UPDATE_CURRENT|PendingIntent.FLAG_IMMUTABLE); if(Build.VERSION.SDK_INT>=31&&!am.canScheduleExactAlarms())am.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP,at,pi);else am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP,at,pi); }
  private void startPet(){ if(!Settings.canDrawOverlays(this)){startActivity(new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:"+getPackageName())));Toast.makeText(this,"请允许‘显示在其他应用上层’，然后回来再点启动",Toast.LENGTH_LONG).show();return;} ContextCompatStart.start(this,new Intent(this,PetOverlayService.class)); }
  static class ContextCompatStart { static void start(Context c,Intent i){ if(Build.VERSION.SDK_INT>=26)c.startForegroundService(i);else c.startService(i);} }
}
