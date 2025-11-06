import sys, asyncio
from datetime import datetime, timedelta
sys.path.append('E:/Projects/Options_probability')
from src.ml.bio_financial.epidemic_data_service import EpidemicDataService

async def main():
    svc = EpidemicDataService()
    end = datetime.now()
    start = end - timedelta(days=365*2)
    df = await svc.get_historical_data(start, end)
    print('LEN', len(df))
    print('INDEX_HEAD', list(df.index[:5]))
    print('INDEX_TAIL', list(df.index[-5:]))
    print('COLUMNS', df.columns.tolist())
    print('HEAD_VOLS', df[['vix','spx_close']].head(3).to_dict('list'))
    print('NAN_COUNTS', df.isna().sum().to_dict())

if __name__ == '__main__':
    asyncio.run(main())

