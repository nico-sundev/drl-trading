# DRL Trading Execution

## Overview
The `drl-trading-execution` package is a core component of the AI Trading monorepo. Its primary responsibility is to facilitate interaction with exchange or broker APIs for executing trades. Additionally, it provides a robust PostgreSQL database API to store and manage trade data efficiently.

## Features
- **Exchange/Broker API Integration**: Seamlessly connect to various trading platforms to execute trades programmatically.
- **Trade Execution**: Automate the process of placing, modifying, and canceling trades.
- **PostgreSQL Database API**: Store trade data, including execution details, status, and historical records, in a structured and reliable manner.
- **Scalability**: Designed to handle high-frequency trading scenarios and large volumes of data.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database
- API credentials for the target exchange or broker

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai_trading.git
   cd drl-trading-execution
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
- Update the configuration file with your exchange/broker API credentials and PostgreSQL connection details.

### Usage
- Import the package and use its API to execute trades and interact with the database.

## Contributing
Contributions are welcome! Please follow the [contribution guidelines](../CONTRIBUTING.md) to submit issues or pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](../LICENSE.txt) file for details.

## Contact
For questions or support, please contact the development team at support@ai_trading.com.
