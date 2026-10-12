import asyncio, json, os, time, sqlite3, secrets, math, collections
from pathlib import Path
from aiohttp import web, ClientSession, WSMsgType
ROOT=Path(__file__).parent
TOKEN=os.getenv('RADAR_TOKEN','')
PAIRS=os.getenv('RADAR_PAIRS','BTC/USD,ETH/USD,SOL/USD,XRP/USD,ADA/USD,DOGE/USD,LINK/USD,AVAX/USD,DOT/USD,LTC/USD').split(',')
PAIRS=[x.strip().upper() for x in PAIRS if x.strip()][:20]
DB=ROOT/'paper.sqlite3'
FEE=.001; MAX_AGE=10
M={p:{'pair':p,'price':None,'ts':0,'bid':None,'ask':None,'bids':{},'asks':{},'trades':collections.deque(maxlen=300),'candles':collections.deque(maxlen=120),'last_trade':0,'last_book':0,'last_candle':0,'spread_pct':None,'imbalance':None,'depth_usd':None,'volume_60s':0,'signal':'WAIT','score':0} for p in PAIRS}
STATUS={'connected':False,'error':'','last_event':0,'reconnects':0}
CFG={'enabled':False,'amount':100,'max_positions':3,'take_profit':2,'stop_loss':1,'max_spread_pct':.2,'min_depth_usd':10000,'min_volume_60s':1000,'min_score':65,'max_daily_loss':100,'cooldown_seconds':300}
lock=asyncio.Lock()
def db():
 c=sqlite3.connect(DB);c.row_factory=sqlite3.Row;return c
def init():
 with db() as c:
  c.execute('create table if not exists balances(kind text primary key, amount real)')
  c.execute('insert or ignore into balances values (?,?)',('spot',7000))
  c.execute('insert or ignore into balances values (?,?)',('futures',3000))
  c.execute('create table if not exists positions(id integer primary key, pair text unique, quantity real, entry real, cost real, opened integer)')
  c.execute('create table if not exists trades(id integer primary key, pair text, pnl real, reason text, closed integer)')
  c.execute('create table if not exists config(id integer primary key, value text)')
  c.execute('insert or ignore into config values(1,?)',(json.dumps(CFG),))
def cfg(c):return json.loads(c.execute('select value from config where id=1').fetchone()[0])
def f(v):
 try:return float(v)
 except:return None
def metrics(m):
 now=time.time();m['trades']=collections.deque((x for x in m['trades'] if now-x[0]<60),maxlen=300)
 m['volume_60s']=round(sum(x[1]*x[2] for x in m['trades']),2)
 bids=sorted(((float(p),float(q)) for p,q in m['bids'].items()),reverse=True)[:20]
 asks=sorted(((float(p),float(q)) for p,q in m['asks'].items()))[:20]
 if bids and asks:
  b=bids[0][0];a=asks[0][0]
  if a<=b:return
  m['bid']=b;m['ask']=a;m['price']=(a+b)/2
  m['spread_pct']=round(100*(a-b)/m['price'],5)
  bv=sum(p*q for p,q in bids);av=sum(p*q for p,q in asks)
  m['depth_usd']=round(bv+av,2);m['imbalance']=round((bv-av)/(bv+av),4) if bv+av else 0
 # Rules are transparent, NOT trained AI or predictive guarantees.
 candles=list(m['candles']); momentum=0
 if len(candles)>=3 and candles[-3][1]>0:momentum=(candles[-1][1]/candles[-3][1]-1)*100
 score=50 + min(25,max(-25,momentum*10)) + min(20,max(-20,(m['imbalance'] or 0)*30))
 if m['spread_pct'] is not None and m['spread_pct']>.2:score-=25
 m['score']=round(max(0,min(100,score)),1)
 m['signal']='WATCH' if m['score']>=65 else 'WAIT'
async def ws_worker(app):
 timeout=__import__('aiohttp').ClientTimeout(total=None)
 async with ClientSession(timeout=timeout) as session:
  while True:
   try:
    async with session.ws_connect('wss://ws.kraken.com/v2',heartbeat=20,timeout=25) as ws:
     STATUS.update(connected=True,error='')
     for channel,extra in [('ticker',{}),('trade',{}),('book',{'depth':25}),('ohlc',{'interval':1})]:
      await ws.send_json({'method':'subscribe','params':{'channel':channel,'symbol':PAIRS,**extra}})
     async for msg in ws:
      if msg.type!=WSMsgType.TEXT:break
      try:data=json.loads(msg.data)
      except:continue
      channel=data.get('channel');items=data.get('data',[])
      if channel not in ('ticker','trade','book','ohlc') or not isinstance(items,list):continue
      async with lock:
       for x in items:
        pair=x.get('symbol');m=M.get(pair)
        if not m:continue
        now=time.time();m['ts']=now;STATUS['last_event']=now
        if channel=='ticker':
         for key in ('bid','ask','last'):
          v=f(x.get(key))
          if v and v>0:
           if key=='last':m['price']=v
           else:m[key]=v
        elif channel=='trade':
         p=f(x.get('price'));q=f(x.get('qty'))
         if p and q and p>0 and q>0:m['trades'].append((now,p,q));m['last_trade']=now;m['price']=p
        elif channel=='ohlc':
         close=f(x.get('close'))
         if close:
          key=x.get('interval_begin') or x.get('timestamp') or str(int(now)//60)
          if m['candles'] and m['candles'][-1][0]==key:m['candles'][-1]=(key,close,float(x.get('volume') or 0))
          else:m['candles'].append((key,close,float(x.get('volume') or 0)))
          m['last_candle']=now
        elif channel=='book':
         if data.get('type')=='snapshot':m['bids'].clear();m['asks'].clear()
         for side in ('bids','asks'):
          book=m[side]
          for row in x.get(side,[]):
           p=f(row.get('price'));q=f(row.get('qty'))
           if p is None or q is None:continue
           if q<=0:book.pop(p,None)
           else:book[p]=q
          if len(book)>100:
           top=sorted(book,reverse=side=='bids')[:25];m[side]={p:book[p] for p in top}
         m['last_book']=now
        metrics(m)
    STATUS['connected']=False;STATUS['reconnects']+=1
   except asyncio.CancelledError:raise
   except Exception as e:STATUS.update(connected=False,error=str(e)[:180]);STATUS['reconnects']+=1
   await asyncio.sleep(5)
async def engine(app):
 while True:
  await asyncio.sleep(1)
  async with lock:
   now=time.time()
   with db() as c:
    conf=cfg(c)
    for p in list(c.execute('select * from positions')):
     m=M.get(p['pair']);price=m['bid'] if m else None
     if not m or not price or now-m['last_book']>MAX_AGE:continue
     pnl=p['quantity']*price*(1-FEE)-p['cost']
     pct=100*pnl/p['cost']
     if pct>=conf['take_profit'] or pct<=-conf['stop_loss']:
      c.execute('update balances set amount=amount+? where kind="spot"',(p['quantity']*price*(1-FEE),))
      c.execute('delete from positions where id=?',(p['id'],))
      c.execute('insert into trades(pair,pnl,reason,closed) values(?,?,?,?)',(p['pair'],pnl,'TP' if pct>=conf['take_profit'] else 'SL',int(now)))
    if not conf['enabled']:continue
    day=int(now)//86400*86400
    daily=c.execute('select coalesce(sum(pnl),0) from trades where closed>=?',(day,)).fetchone()[0]
    if daily<=-conf['max_daily_loss']:
     conf['enabled']=False;c.execute('update config set value=? where id=1',(json.dumps(conf),));continue
    active={r['pair'] for r in c.execute('select pair from positions')}
    if len(active)>=conf['max_positions']:continue
    for pair,m in sorted(M.items(),key=lambda v:v[1]['score'],reverse=True):
     if len(active)>=conf['max_positions']:break
     if pair in active or not STATUS['connected'] or not m['ask'] or now-m['last_book']>MAX_AGE or now-m['last_trade']>MAX_AGE or now-m['last_candle']>120:continue
     if m['score']<conf['min_score'] or m['spread_pct'] is None or m['spread_pct']>conf['max_spread_pct'] or (m['depth_usd'] or 0)<conf['min_depth_usd'] or m['volume_60s']<conf['min_volume_60s']:continue
     last=c.execute('select max(closed) from trades where pair=?',(pair,)).fetchone()[0]
     if last and now-last<conf['cooldown_seconds']:continue
     balance=c.execute('select amount from balances where kind="spot"').fetchone()[0]
     amount=conf['amount']
     if balance<amount:continue
     # Simulated market buy at best ASK with fee; no invented execution guarantee.
     quantity=(amount*(1-FEE))/m['ask']
     c.execute('update balances set amount=amount-? where kind="spot"',(amount,))
     c.execute('insert into positions(pair,quantity,entry,cost,opened) values(?,?,?,?,?)',(pair,quantity,m['ask'],amount,int(now)))
     active.add(pair)
def authorized(request):return bool(TOKEN) and secrets.compare_digest(request.headers.get('Authorization',''),'Bearer '+TOKEN)
@web.middleware
async def guard(request,handler):
 if request.path.startswith('/api/') and not authorized(request):return web.json_response({'error':'دسترسی غیرمجاز'},status=401)
 return await handler(request)
async def index(request):return web.FileResponse(ROOT/'index.html')
async def state(request):
 async with lock:
  with db() as c:
   positions=[dict(r) for r in c.execute('select * from positions')]
   for p in positions:
    m=M.get(p['pair']);bid=m['bid'] if m else None
    p['pnl']=round(p['quantity']*bid*(1-FEE)-p['cost'],3) if bid and time.time()-m['last_book']<=MAX_AGE else None
   markets=[]
   for pair,m in M.items():
    markets.append({k:(list(v)[-10:] if isinstance(v,collections.deque) else v) for k,v in m.items() if k not in ('bids','asks','trades')})
   return web.json_response({'markets':markets,'status':STATUS,'config':cfg(c),'balances':{r['kind']:r['amount'] for r in c.execute('select * from balances')},'positions':positions,'trades':[dict(r) for r in c.execute('select * from trades order by closed desc limit 100')],'server_time':time.time()})
async def configure(request):
 try:
  data=await request.json()
  async with lock:
   with db() as c:
    conf=cfg(c)
    for k in conf:
     if k in data:conf[k]=data[k]
    ranges={'amount':(1,7000),'max_positions':(1,20),'take_profit':(.1,50),'stop_loss':(.1,50),'max_spread_pct':(.001,5),'min_depth_usd':(0,1e9),'min_volume_60s':(0,1e9),'min_score':(1,100),'max_daily_loss':(1,7000),'cooldown_seconds':(0,86400)}
    for k,(lo,hi) in ranges.items():
     v=conf[k]
     if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('مقدار نامعتبر: '+k)
    if type(conf['enabled']) is not bool:raise ValueError('enabled نامعتبر')
    conf['max_positions']=int(conf['max_positions']);conf['cooldown_seconds']=int(conf['cooldown_seconds'])
    c.execute('update config set value=? where id=1',(json.dumps(conf),))
  return web.json_response({'ok':True})
 except (ValueError,TypeError) as e:return web.json_response({'error':str(e)},status=400)
async def close(request):
 data=await request.json()
 async with lock:
  with db() as c:
   p=c.execute('select * from positions where id=?',(data.get('id'),)).fetchone()
   if not p:return web.json_response({'error':'موقعیت پیدا نشد'},status=404)
   m=M.get(p['pair']);now=time.time()
   if not m or not m['bid'] or now-m['last_book']>MAX_AGE:return web.json_response({'error':'دفتر سفارش تازه نیست؛ خروج با قیمت قدیمی ممنوع'},status=409)
   value=p['quantity']*m['bid']*(1-FEE);pnl=value-p['cost']
   c.execute('update balances set amount=amount+? where kind="spot"',(value,))
   c.execute('delete from positions where id=?',(p['id'],))
   c.execute('insert into trades(pair,pnl,reason,closed) values(?,?,?,?)',(p['pair'],pnl,'MANUAL',int(now)))
 return web.json_response({'ok':True})
async def start(app):
 if not TOKEN:raise RuntimeError('RADAR_TOKEN is required')
 init();app['tasks']=[asyncio.create_task(ws_worker(app)),asyncio.create_task(engine(app))]
async def stop(app):
 for t in app['tasks']:t.cancel()
 await asyncio.gather(*app['tasks'],return_exceptions=True)
app=web.Application(middlewares=[guard]);app.router.add_get('/',index);app.router.add_get('/api/state',state);app.router.add_post('/api/config',configure);app.router.add_post('/api/close',close);app.on_startup.append(start);app.on_cleanup.append(stop)
if __name__=='__main__':web.run_app(app,host=os.getenv('HOST','127.0.0.1'),port=int(os.getenv('PORT','8080')))
