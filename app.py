import json, os, sqlite3, threading, time, urllib.request, urllib.error
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

DB=os.environ.get('RADAR_DB','radar.db')
HOST=os.environ.get('HOST','0.0.0.0'); PORT=int(os.environ.get('PORT','8000'))
LOCK=threading.RLock()
DEFAULT={'enabled':False,'mode':'spot','trade_usdt':100,'leverage':2,'tp_pct':3,'sl_pct':1.5,'signal_24h_pct':4,'max_positions':5,'interval_sec':300}
EXCLUDED={'tether','usd-coin','dai','first-digital-usd','usds','ethena-usde','paypal-usd','true-usd','usdd','wrapped-steth','weth','wrapped-bitcoin','coinbase-wrapped-btc','staked-ether','wrapped-eeth','renzo-restaked-eth'}
FEE=0.001; SLIPPAGE=0.0005

def db():
 c=sqlite3.connect(DB,timeout=30);c.row_factory=sqlite3.Row;return c

def init():
 with db() as c:
  c.execute('CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)')
  c.execute('INSERT OR IGNORE INTO settings VALUES (1,?)',(json.dumps(DEFAULT),))
  c.execute('CREATE TABLE IF NOT EXISTS balances (mode TEXT PRIMARY KEY, cash REAL NOT NULL)')
  c.execute("INSERT OR IGNORE INTO balances VALUES ('spot',7000)")
  c.execute("INSERT OR IGNORE INTO balances VALUES ('futures',3000)")
  c.execute('CREATE TABLE IF NOT EXISTS positions (id INTEGER PRIMARY KEY AUTOINCREMENT,coin_id TEXT,symbol TEXT,mode TEXT,margin REAL,leverage REAL,entry REAL,opened_at TEXT)')
  c.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT,coin_id TEXT,symbol TEXT,mode TEXT,margin REAL,leverage REAL,entry REAL,exit REAL,pnl REAL,reason TEXT,opened_at TEXT,closed_at TEXT)')
  c.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)')

def settings(c):return json.loads(c.execute('SELECT value FROM settings WHERE id=1').fetchone()['value'])
def stamp():return datetime.now(timezone.utc).isoformat(timespec='seconds')
def setmeta(c,k,v):c.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(k,str(v)))
def market():
 url='https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false&price_change_percentage=24h'
 req=urllib.request.Request(url,headers={'User-Agent':'CryptoRadarPaperEngine/1.0','Accept':'application/json'})
 with urllib.request.urlopen(req,timeout=25) as r:data=json.load(r)
 if not isinstance(data,list) or len(data)<20:raise ValueError('Market feed incomplete')
 return {x['id']:x for x in data if x.get('id') and isinstance(x.get('current_price'),(int,float)) and x['current_price']>0}

def pnl(p,price):
 n=p['margin']*p['leverage']; exit_price=price*(1-SLIPPAGE)
 return n*(exit_price/p['entry']-1)-n*FEE*2

def close(c,p,price,reason):
 gain=pnl(p,price)
 # Isolated paper margin: losses capped to deposited margin (simplified liquidation)
 gain=max(-p['margin'],gain)
 c.execute('UPDATE balances SET cash=cash+? WHERE mode=?',(max(0,p['margin']+gain),p['mode']))
 c.execute('INSERT INTO trades(coin_id,symbol,mode,margin,leverage,entry,exit,pnl,reason,opened_at,closed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(p['coin_id'],p['symbol'],p['mode'],p['margin'],p['leverage'],p['entry'],price*(1-SLIPPAGE),gain,reason,p['opened_at'],stamp()))
 c.execute('DELETE FROM positions WHERE id=?',(p['id'],))

def scan(force=False):
 prices=market()
 with LOCK,db() as c:
  s=settings(c)
  if not s['enabled'] and not force:return {'message':'ربات متوقف است'}
  closed=[];opened=[]
  for p in c.execute('SELECT * FROM positions').fetchall():
   coin=prices.get(p['coin_id'])
   if not coin:continue
   change=(coin['current_price']/p['entry']-1)*100*p['leverage']
   if change>=s['tp_pct'] or change<=-s['sl_pct']:
    reason='TP' if change>=s['tp_pct'] else 'SL';close(c,p,coin['current_price'],reason);closed.append(p['symbol']+' '+reason)
  active=c.execute('SELECT COUNT(*) AS n FROM positions').fetchone()['n']
  candidates=sorted((v for v in prices.values() if v['id'] not in EXCLUDED and isinstance(v.get('price_change_percentage_24h'),(int,float)) and v['price_change_percentage_24h']>=s['signal_24h_pct']),key=lambda x:x['price_change_percentage_24h'],reverse=True)
  for coin in candidates:
   if active>=s['max_positions']:break
   if c.execute('SELECT 1 FROM positions WHERE coin_id=?',(coin['id'],)).fetchone():continue
   mode=s['mode'];amount=s['trade_usdt'];lev=1 if mode=='spot' else s['leverage']
   bal=c.execute('SELECT cash FROM balances WHERE mode=?',(mode,)).fetchone()['cash']
   if bal<amount:break
   entry=coin['current_price']*(1+SLIPPAGE)
   c.execute('UPDATE balances SET cash=cash-? WHERE mode=?',(amount,mode))
   c.execute('INSERT INTO positions(coin_id,symbol,mode,margin,leverage,entry,opened_at) VALUES(?,?,?,?,?,?,?)',(coin['id'],coin['symbol'].upper(),mode,amount,lev,entry,stamp()))
   opened.append(coin['symbol'].upper());active+=1
  setmeta(c,'last_scan',stamp());setmeta(c,'last_error','')
  return {'opened':opened,'closed':closed,'market_count':len(prices)}

def snapshot():
 with LOCK,db() as c:
  s=settings(c); balances={r['mode']:r['cash'] for r in c.execute('SELECT * FROM balances')}
  pos=[dict(x) for x in c.execute('SELECT * FROM positions ORDER BY id DESC')]
  trades=[dict(x) for x in c.execute('SELECT * FROM trades ORDER BY id DESC LIMIT 50')]
  meta={r['key']:r['value'] for r in c.execute('SELECT * FROM meta')}
  return {'settings':s,'balances':balances,'positions':pos,'trades':trades,'meta':meta}

def worker():
 while True:
  try:
   with LOCK,db() as c:s=settings(c)
   if s['enabled']:
    try:scan()
    except Exception as e:
     with LOCK,db() as c:setmeta(c,'last_error',str(e)[:250])
   time.sleep(max(60,int(s['interval_sec'])))
  except Exception:time.sleep(60)

HTML='''<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Crypto Signal Radar AI</title><style>body{background:#0b1221;color:#edf3ff;font-family:Tahoma,Arial;margin:0;padding:15px}main{max-width:950px;margin:auto}h1{font-size:23px}.card{background:#162238;border:1px solid #33465f;border-radius:14px;padding:15px;margin:12px 0}input,select,button{background:#253958;color:white;border:1px solid #526b8a;border-radius:8px;padding:10px;font-size:15px;max-width:100%}input{width:95px}button{cursor:pointer;background:#176e92}button.stop{background:#914050}label{display:inline-flex;flex-direction:column;gap:7px;margin:6px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:9px;border-bottom:1px solid #40506a;text-align:right} .scroll{overflow:auto}.muted{color:#b6c5da}.good{color:#6fe3ac}.bad{color:#ff9393}</style><main><h1>📡 Crypto Signal Radar AI</h1><p class="muted">موتور معاملات آزمایشی — بدون اتصال به صرافی و بدون پول واقعی</p><div class="card"><b id="status">در حال بارگذاری...</b><p id="meta" class="muted"></p><button onclick="control('start')">▶ شروع خودکار</button> <button class="stop" onclick="control('stop')">■ توقف</button> <button onclick="control('scan')">🔎 اسکن فوری</button><p id="notice"></p></div><div class="card"><h3>موجودی آزاد آزمایشی</h3><div id="balances"></div></div><div class="card"><h3>تنظیمات</h3><label>بازار<select id="mode"><option value="spot">اسپات</option><option value="futures">فیوچرز شبیه‌سازی‌شده</option></select></label><label>مبلغ هر معامله USDT<input id="trade_usdt" type="number" min="1" max="3000"></label><label>اهرم<input id="leverage" type="number" min="1" max="10"></label><label>حد سود %<input id="tp_pct" type="number" min="0.1" max="100" step="0.1"></label><label>حد ضرر %<input id="sl_pct" type="number" min="0.1" max="100" step="0.1"></label><label>حداقل رشد ۲۴ساعته %<input id="signal_24h_pct" type="number" min="0" max="100" step="0.1"></label><label>حداکثر موقعیت<input id="max_positions" type="number" min="1" max="20"></label><label>فاصله بررسی (ثانیه)<input id="interval_sec" type="number" min="60" max="3600"></label><p><button onclick="save()">💾 ذخیره تنظیمات</button></p></div><div class="card"><h3>موقعیت‌های باز</h3><div class="scroll"><table><thead><tr><th>ارز</th><th>بازار</th><th>وثیقه</th><th>اهرم</th><th>قیمت ورود</th><th>عملیات</th></tr></thead><tbody id="positions"></tbody></table></div></div><div class="card"><h3>۵۰ معامله بسته‌شده اخیر</h3><div class="scroll"><table><thead><tr><th>ارز</th><th>بازار</th><th>سود/زیان USDT</th><th>دلیل</th><th>زمان</th></tr></thead><tbody id="trades"></tbody></table></div></div><p class="muted">هشدار: قیمت‌ها از CoinGecko و با تأخیر احتمالی دریافت می‌شوند. این مدل ساده، اجرای واقعی سفارش، لیکوییدیشن، نرخ تأمین مالی و لغزش واقعی بازار را بازسازی نمی‌کند. سرور باید روشن بماند؛ میزبانی رایگان ممکن است متوقف شود.</p></main><script>
let first=true;const el=x=>document.getElementById(x);const fmt=x=>Number(x).toLocaleString('en-US',{maximumFractionDigits:4});
async function api(path,data){let o=data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)};let r=await fetch(path,o),j=await r.json();if(!r.ok)throw Error(j.error||r.status);return j}
async function load(){try{let d=await api('/api/state'),s=d.settings;el('status').textContent=s.enabled?'🟢 موتور روشن است':'🔴 موتور متوقف است';el('meta').textContent='آخرین بررسی: '+(d.meta.last_scan||'هنوز انجام نشده')+(d.meta.last_error?' | خطا: '+d.meta.last_error:'');el('balances').textContent='اسپات: '+fmt(d.balances.spot)+' USDT | فیوچرز: '+fmt(d.balances.futures)+' USDT';if(first){for(let k in s)if(el(k))el(k).value=s[k];first=false}el('positions').innerHTML=d.positions.map(p=>`<tr><td>${p.symbol}</td><td>${p.mode}</td><td>${fmt(p.margin)}</td><td>${p.leverage}×</td><td>${fmt(p.entry)}</td><td><button onclick="closeTrade(${p.id})">بستن</button></td></tr>`).join('')||'<tr><td colspan="6">موقعیت بازی نیست</td></tr>';el('trades').innerHTML=d.trades.map(t=>`<tr><td>${t.symbol}</td><td>${t.mode}</td><td class="${t.pnl>=0?'good':'bad'}">${fmt(t.pnl)}</td><td>${t.reason}</td><td>${t.closed_at}</td></tr>`).join('')||'<tr><td colspan="5">هنوز معامله بسته‌شده‌ای نیست</td></tr>'}catch(e){el('notice').textContent='خطا: '+e.message}}
async function control(action){try{el('notice').textContent='در حال انجام...';let r=await api('/api/'+action,{});el('notice').textContent=JSON.stringify(r);await load()}catch(e){el('notice').textContent='خطا: '+e.message}}
async function save(){let d={};for(let k of ['mode','trade_usdt','leverage','tp_pct','sl_pct','signal_24h_pct','max_positions','interval_sec'])d[k]=k==='mode'?el(k).value:Number(el(k).value);try{await api('/api/settings',d);el('notice').textContent='تنظیمات ذخیره شد';first=true;load()}catch(e){el('notice').textContent=e.message}}
async function closeTrade(id){if(!confirm('این موقعیت آزمایشی بسته شود؟'))return;try{await api('/api/close',{id});load()}catch(e){el('notice').textContent=e.message}}
load();setInterval(load,15000)</script></html>'''
class Handler(BaseHTTPRequestHandler):
 def reply(self,code,obj):
  b=json.dumps(obj,ensure_ascii=False).encode();self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self):
  if urlparse(self.path).path=='/':
   b=HTML.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
  elif self.path=='/api/state':self.reply(200,snapshot())
  elif self.path=='/health':self.reply(200,{'ok':True})
  else:self.reply(404,{'error':'not found'})
 def do_POST(self):
  try:
   size=int(self.headers.get('Content-Length','0'))
   if size>10000:raise ValueError('Request too large')
   data=json.loads(self.rfile.read(size) or b'{}');path=urlparse(self.path).path
   if path=='/api/settings':
    with LOCK,db() as c:
     s=settings(c);s.update({k:data[k] for k in DEFAULT if k in data and k!='enabled'})
     if s['mode'] not in ('spot','futures'):raise ValueError('Invalid mode')
     for k,lo,hi in [('trade_usdt',1,3000),('leverage',1,10),('tp_pct',.1,100),('sl_pct',.1,100),('signal_24h_pct',0,100),('max_positions',1,20),('interval_sec',60,3600)]:
      v=float(s[k]);
      if not lo<=v<=hi:raise ValueError('Invalid '+k)
     for k in ('leverage','max_positions','interval_sec'):s[k]=int(s[k])
     c.execute('UPDATE settings SET value=? WHERE id=1',(json.dumps(s),))
    self.reply(200,{'ok':True})
   elif path in ('/api/start','/api/stop'):
    with LOCK,db() as c:
     s=settings(c);s['enabled']=path.endswith('start');c.execute('UPDATE settings SET value=? WHERE id=1',(json.dumps(s),))
    self.reply(200,{'enabled':s['enabled']})
   elif path=='/api/scan':self.reply(200,scan(force=True))
   elif path=='/api/close':
    prices=market()
    with LOCK,db() as c:
     p=c.execute('SELECT * FROM positions WHERE id=?',(int(data['id']),)).fetchone()
     if not p:raise ValueError('Position not found')
     if p['coin_id'] not in prices:raise ValueError('Price unavailable: position left open')
     close(c,p,prices[p['coin_id']]['current_price'],'MANUAL')
    self.reply(200,{'ok':True})
   else:self.reply(404,{'error':'not found'})
  except Exception as e:self.reply(400,{'error':str(e)[:250]})
if __name__=='__main__':
 init();threading.Thread(target=worker,daemon=True).start();print(f'Paper dashboard http://localhost:{PORT}',flush=True);ThreadingHTTPServer((HOST,PORT),Handler).serve_forever()
