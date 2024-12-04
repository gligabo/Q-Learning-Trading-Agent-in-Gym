import yfinance as yf
import pandas as pd

class GetData:
    def from_yfinance( self, ticker, start = None, end = None, interval = "1d" ):
        data = yf.download( ticker, start = start, end = end, interval = interval )
        return data
    

class CleanData:
    def cleaning( self, data ):

        if 'Adj Close' in data.columns:
            data.drop( 'Adj Close', axis=1, inplace=True )
        
        data.rename( columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True )

        if isinstance( data.columns, pd.MultiIndex ):
            data.columns = data.columns.get_level_values( 0 )

        data.reset_index( inplace=True )

        if 'Date' in data.columns:
            data.rename( columns = {"Date": "date"}, inplace=True )
        elif 'date' not in data.columns:
            raise KeyError( "A coluna 'Date' ou 'date' não foi encontrada no DataFrame." )

        data["date"] = pd.to_datetime(data["date"], errors='coerce')

        if data["date"].isnull().any():
            raise ValueError( "Existem valores NaT na coluna 'date' após a conversão para datetime." )

        data.set_index( "date", inplace=True )

        if data.index.tz is not None:
            data.index = data.index.tz_localize( None )

        return data
