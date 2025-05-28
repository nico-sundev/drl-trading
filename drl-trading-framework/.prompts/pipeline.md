## Pipeline
### Loading the data
- the raw data is structured as OHLCV timeseries data and locally stored like that:
    - symbol 1
        - OHLCV dataset timeframe 1
        - OHLCV dataset timeframe ...
        - OHLCV dataset timeframe n
    - symbol n
        - OHLCV datasets ...
- All timeseries data from different timeframes are loaded
- the lowest timeframe´s dataset determines called the `base dataset`
- all higher timeframe´s datasets are called `other datasets`
- Responsible classes: mainly CsvDataImportService and DataImportManager

### Stripping other datasets
- to save computing power, uneccesary data from higher timeframes will be removed
- the last timestamp in base dataset will serve as the threshold timestamp and all rows of other dataset will be removed if they are after
- Responsible classes: TimeframeStripperService

### feature computing
- Serves all computed features for a given timeframes dataset and symbol
- Feast Feature store is being used as a cache
- Every feature has subfeatures, which are defined inside every Feature class (extending BaseFeature):
    - Example: MacdFeature
        - macd_cross_bullish
        - macd_cross_bearish
        - macd_trend
- At the very first call, a feature store for the given symbol and timeframe dataset is created
- A feature view for every feature is created, if not yet existing
- A feature is computed if not found in the store
- Responsible classes: FeatureClassFactoryInterface, FeatureConfigRegistry and mostly FeatureAggregator

### Merging the timeframes
- Every timeframes dataset is then compared to the base dataset
- Every record of the base dataset is pointing to the last confirmed candles feature value
- The basic idea is: If a human would look at a specific lower timeframe close candle, the feature value of the last closed and confirmed higher timeframe candle is being taken into account
- No future sight!
- Responsible classes: MergingService

### Splitting the final dataset
- The final training dataset will then be split into three parts:
    - training dataset
    - validation dataset
    - test dataset
- Responsible classes: SplitService

### Start training
- create environments, instantiate agents
- Responsible classes: CustomEnv, AgentFactory, AgentTrainingService

## Bootstrap
- The engine is being started here in `bootstrap.py`

## configuration
- a file `applicationConfig.json` reflecting `application_config.py` module is necessary to bootstrap
