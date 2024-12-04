import ta
import pandas as pd
import numpy as np

class AddIndicators:
    
    def add_indicator( self, data, indicator_name, **kwargs ):
        """
        Adiciona indicadores técnicos ao dataframe.
        
        Args:
            data (pd.DataFrame): O dataframe com os dados de mercado (deve conter colunas 'close', 'high', 'low', etc).
            indicator_name (str): Nome do indicador técnico a ser adicionado (ex.: 'rsi', 'macd', 'sma', 'bollinger').
            **kwargs: Parâmetros necessários para inicializar o indicador técnico.
        
        Returns:
            pd.DataFrame: Dataframe atualizado com as colunas do indicador técnico adicionado.
        """

        indicator_mapping = {
            "rsi": ta.momentum.RSIIndicator,
            "macd": ta.trend.MACD,
            "sma": ta.trend.SMAIndicator,
            "bollinger": ta.volatility.BollingerBands
        }

        if indicator_name not in indicator_mapping:
            raise ValueError(
                f"Indicador '{indicator_name}' não é suportado. "
                f"Indicadores disponíveis: {list( indicator_mapping.keys() )}"
            )

        indicator = indicator_mapping[indicator_name]( **kwargs )

        if indicator_name == "macd":
            data["macd"] = indicator.macd()
            data["macd_signal"] = indicator.macd_signal()
            data["macd_diff"] = indicator.macd_diff()

        elif indicator_name == "rsi":
            data["rsi"] = indicator.rsi()

        elif indicator_name == "sma":
            data["sma"] = indicator.sma_indicator()

        elif indicator_name == "bollinger":
            data["bollinger_mavg"] = indicator.bollinger_mavg()
            data["bollinger_hband"] = indicator.bollinger_hband()
            data["bollinger_lband"] = indicator.bollinger_lband()
            data["bollinger_width"] = indicator.bollinger_wband()

        return data

class Discretize:
    
    def __init__( self, num_states = 10, window = None ):
        """
        Inicializa a classe para aplicar discretização de indicadores técnicos.

        Args:
            num_states (int): Número de estados discretos para os valores contínuos.
        """

        self.num_states = num_states
        self.window = window


    def calculate_zscore( self, series ):
        """
        Calcula o z-score para uma série.

        Args:
            series (pd.Series): Série de valores contínuos.

        Returns:
            pd.Series: Z-scores da série.
        """

        zscore = ( series - series.mean() ) / series.std()
        return zscore.fillna(0)


    def rolling_discretize( self, values ):
        """
        Aplica discretização em janelas deslizantes.

        Args:
            values (pd.Series): Série de valores contínuos.

        Returns:
            pd.Series: Série discretizada com base nas janelas deslizantes.
        """
        def apply_qcut( x ):
            if len( x.dropna() ) < self.num_states:
                return np.nan
            try:
                discretized = pd.qcut( x, q=self.num_states, labels=range( self.num_states ), duplicates='drop' )
                return discretized.iloc[-1]
            except ValueError:
                return np.nan

        discretized_series = values.rolling( window=self.window, min_periods=self.num_states ).apply( apply_qcut, raw=False )
        discretized_series = discretized_series.fillna( 0 ).astype( int )
        return discretized_series



    def discretize_states( self, values, num_states ):
        """
        Converte valores contínuos em estados discretos com base em quantis.

        Args:
            values (pd.Series): Valores contínuos para discretizar.
            num_states (int): Número de estados discretos (blocos de quantis).

        Returns:
            pd.Series: Valores convertidos para estados discretos.
        """

        if self.window:
            return values.rolling( self.window, min_periods=self.num_states ).apply(
                self.rolling_discretizer, raw=False
            )
        else:
            quantiles = values.quantile( [i / self.num_states for i in range( 1, self.num_states )] ).values
            bins = [-float('inf')] + list( quantiles ) + [float('inf')]

            discrete_values = pd.cut( values, bins = bins, labels = range( num_states ) )
            discrete_values = discrete_values.astype( float ).fillna( 0 ).astype( int )
        
        return discrete_values

    def process_indicator( self, data, column_name, base_name ):
        """
        Processa uma coluna de indicador: calcula z-score, lida com NaNs e discretiza.

        Args:
            data (pd.DataFrame): DataFrame contendo os dados.
            column_name (str): Nome da coluna do indicador a ser processado.
            base_name (str): Base do nome para as novas colunas criadas.

        Returns:
            pd.DataFrame: DataFrame atualizado com as novas colunas processadas.
        """

        feature_prefix = f"feature_{base_name}"

        zscore_column = f"{base_name}_zscore"
        data[zscore_column] = self.calculate_zscore( data[column_name] )

        discrete_column = f"{feature_prefix}_discrete"

        if self.window:
            data[discrete_column] = self.rolling_discretize( data[zscore_column] )
        else:
            data[discrete_column] = self.discretize_states( data[zscore_column] )

        return data