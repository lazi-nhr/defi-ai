### **Binance Perpetual Futures Funding Rate Data Report**

**Report Generation Date:** October 1, 2025

**Data Source:** Official Binance API (Binance USDⓈ-M Futures)

**Data Coverage Period:** August 22, 2024, to August 22, 2025

#### **1. Report Overview**

This report provides a detailed explanation of the collection methodology, file structure, field descriptions, and interpretation of key metrics for the provided dataset. The dataset contains the complete historical funding rates for all active perpetual contracts on the Binance USDⓈ-M (USDⓈ-Margined) market within the specified period. This data is highly valuable for market sentiment analysis, trading strategy backtesting, and understanding the costs associated with the derivatives market.

#### **2. Data Collection Methodology**

The data was programmatically collected from the official Binance API using a custom Python script, ensuring accuracy, completeness, and reproducibility.

- **Collection Tool:** Python 3.10+
    
- **Core Libraries:** `python-binance`, `pandas`
    
- **Collection Process:**
    
    1. **Initialization:** Upon execution, the script connects to the public API endpoint for Binance USDⓈ-M perpetual futures.
        
    2. **Symbol List Retrieval:** The script automatically fetches a list of all perpetual contracts that are actively in a "TRADING" state on the Binance market at the time of execution.
        
    3. **Iterative Fetching with Pagination:** The script iterates through each perpetual contract, using the specified date range (August 22, 2024, to August 22, 2025) as the time window. It repeatedly calls the historical funding rate API endpoint. As the API has a limit on the number of records per request (1000), the script includes built-in pagination logic to ensure all records within the time frame are retrieved.
        
    4. **Data Storage:** The complete historical funding rate data for each trading pair is cleaned, structured, and saved as an individual `.parquet` file.
        

#### **3. Data File Structure and Naming Convention**

All data files are located in the `data/binance_funding/` directory.

- **File Format:** **Parquet**. This is a highly efficient columnar storage format that offers faster read speeds, smaller file sizes, and self-contained data type information compared to formats like CSV.
    
- **File Naming Convention:** `SYMBOL_funding_STARTDATE_ENDDATE.parquet`
    
    - **Example:** `BTC_funding_2024-08-22_2025-08-22.parquet`
        
- Guide to Understanding Trading Symbols:
    
    To facilitate trading for assets with extremely low unit prices, Binance employs a multiple-contract naming system. Understanding these symbols is key to using the data correctly:
    

|   |   |   |   |
|---|---|---|---|
|**Example Symbol in Filename**|**Corresponding Real Contract**|**Meaning**|**Description**|
|`ADA`|`ADA/USDT`|**Standard Contract**: 1 contract unit = 1 ADA coin.|Used for mainstream assets with a moderate price.|
|`1000LUNC`|`1000LUNC/USDT`|**1000x Multiple Contract**: 1 contract unit = 1000 LUNC coins.|Used for assets with a very low unit price to simplify pricing and trading.|
|`1MBABYDOGE`|`1000000BABYDOGE/USDT`|**Millionx Multiple Contract**: 1 contract unit = 1,000,000 BABYDOGE coins.|Used for assets with an even lower unit price. `M` stands for Million.|
|`AAVEUSDC`|`AAVE/USDC`|**USDC-Margined Contract**: Settlement is done using the USDC stablecoin.|An alternative to the default USDT-margined contracts.|
|`1000SHIBUSDC`|`1000SHIB/USDC`|**Composite Contract**: A 1000x multiple contract that is also settled in USDC.|Combines the features of a multiple contract and a USDC-margined contract.|

#### **4. Data Field Descriptions**

Each Parquet file contains the following core fields:

|   |   |   |
|---|---|---|
|**Field Name**|**Data Type**|**Description**|
|`symbol`|`string`|The official Binance trading pair symbol, e.g., `BTCUSDT`, `1000LUNCUSDT`.|
|`fundingTime`|`datetime64[ms]`|The precise timestamp (in UTC) when the funding fee was settled. This typically occurs every 8 hours.|
|`fundingRate`|`float`|**Key Metric: The Funding Rate**. This is a percentage representing the payment ratio between long and short positions.|
|`markPrice`|`float`|The mark price at the moment of funding settlement, used for calculating unrealized PnL.|

#### **5. Interpretation of the Key Metric: Funding Rate**

The funding rate is the core mechanism of perpetual contracts, designed to keep the contract's price closely anchored to the corresponding spot price. It is not a fee charged by the exchange but a direct payment exchanged between users holding long and short positions.

- **Positive Funding Rate:**
    
    - **Scenario:** Typically occurs when market sentiment is bullish, and the perpetual contract price is higher than the spot price.
        
    - **Mechanism:** **Traders with long positions pay a funding fee to traders with short positions.**
        
    - **Data Application:** A sustained positive funding rate indicates strong demand for leveraged long positions in the market.
        
- **Negative Funding Rate:**
    
    - **Scenario:** Typically occurs during market panic or strong bearish sentiment, when the perpetual contract price is lower than the spot price.
        
    - **Mechanism:** **Traders with short positions pay a funding fee to traders with long positions.**
        
    - **Data Application:** A sustained negative funding rate indicates strong demand for shorting or hedging in the market.
        
- **How to Calculate the Funding Fee:**
    
    - `Funding Fee = Position Nominal Value × Funding Rate`
        
    - Position Nominal Value = Mark Price × Position Size × (Contract Face Value, which is 1 for standard contracts)
        

#### **6. Usage Recommendations and Notes**

- **Data Timeliness:** This dataset is a snapshot covering the period from August 22, 2024, to August 22, 2025. Binance may list new contracts or delist old ones; this dataset only includes contracts that were active at the time of data collection.
    
- **Data Accuracy:** The data is sourced directly from the official Binance API without any modification, only format conversion.
    
- **Strategy Applications:** This data can be used to build market sentiment indicators, develop arbitrage strategies (e.g., spot-futures arbitrage, funding rate arbitrage), or accurately calculate position holding costs and returns.
    
- **Official Documentation:** For the most authoritative definitions and calculation details, please refer to the official Binance API documentation.