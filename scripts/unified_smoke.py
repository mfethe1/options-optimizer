import urllib.request, urllib.parse, json
from statistics import mean

BASE='http://127.0.0.1:8017'

def fetch(symbol, time_range='1Y'):
    q=urllib.parse.urlencode({'symbol':symbol,'time_range':time_range})
    req=urllib.request.Request(f"{BASE}/unified/forecast/all?{q}", method='POST')
    with urllib.request.urlopen(req, timeout=60) as r:
        data=json.loads(r.read().decode('utf-8'))
        tl=data.get('timeline', [])
        acts=[p.get('actual') for p in tl if isinstance(p.get('actual'), (int,float))]
        times=[p.get('time') for p in tl]
        return tl, acts, times

def sma(arr, n):
    if len(arr) < n: return None
    return sum(arr[-n:])/n

def rsi(values, period=14):
    if len(values) <= period: return None
    gains=0.0; losses=0.0
    for i in range(1, period+1):
        d = values[i]-values[i-1]
        gains += max(d,0)
        losses += -min(d,0)
    avg_gain=gains/period
    avg_loss=losses/period
    if avg_loss==0: return 100.0
    rs=avg_gain/avg_loss
    r = 100 - (100/(1+rs))
    for i in range(period+1, len(values)):
        d=values[i]-values[i-1]
        gain=max(d,0); loss=-min(d,0)
        avg_gain=(avg_gain*(period-1)+gain)/period
        avg_loss=(avg_loss*(period-1)+loss)/period
        rs=avg_gain/avg_loss if avg_loss!=0 else float('inf')
        r = 100 - (100/(1+rs))
    return r

symbols=[('^VIX','VIX'), ('NVDA','NVDA'), ('PLTR','PLTR')]
for sy, label in symbols:
    try:
        tl,acts,times=fetch(sy, '1Y')
        last=acts[-1] if acts else None
        out={
            'symbol': label,
            'points': len(acts),
            'first': times[:2],
            'last': times[-2:],
            'min': min(acts) if acts else None,
            'max': max(acts) if acts else None,
            'last_price': round(last,2) if last is not None else None,
            'sma20': round(sma(acts,20),2) if sma(acts,20) else None,
            'sma50': round(sma(acts,50),2) if sma(acts,50) else None,
            'sma200': round(sma(acts,200),2) if sma(acts,200) else None,
            'rsi14': round(rsi(acts,14),1) if rsi(acts,14) else None,
        }
        print(out)
    except Exception as e:
        print({'symbol': label, 'error': str(e)})

