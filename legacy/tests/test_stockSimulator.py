import unittest
from unittest.mock import patch, MagicMock
from app import stockSimulator
import pandas as pd

class TestStockSimulator(unittest.TestCase):

    @patch('app.stockSimulator.sqlite3.connect')
    def test_initialize_portfolio_db_creates_tables_and_initializes(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        # Simulate no portfolio_state rows
        mock_cursor.fetchone.return_value = [0]
        stockSimulator.initialize_portfolio_db()
        self.assertTrue(mock_cursor.execute.called)
        self.assertTrue(mock_conn.commit.called)
        self.assertTrue(mock_conn.close.called)

    @patch('app.stockSimulator.db_manager.get_portfolio_state')
    @patch('app.stockSimulator.get_current_stock_prices')
    def test_calculate_current_portfolio_value_with_cash(self, mock_get_prices, mock_get_state):
        # Simulate holdings and prices
        mock_get_state.return_value = (None, [(1, 'AAPL', 2, 100.0, 200.0)])
        mock_get_prices.return_value = {'AAPL': 150.0}
        value = stockSimulator.calculate_current_portfolio_value_with_cash(500)
        self.assertEqual(value, 500 + 2 * 150.0)

    @patch('app.stockSimulator.db_manager.get_portfolio_state')
    @patch('app.stockSimulator.get_current_stock_prices')
    def test_calculate_current_portfolio_value(self, mock_get_prices, mock_get_state):
        mock_get_state.return_value = ([1, 1000, 0, None, None], [(1, 'AAPL', 2, 100.0, 200.0)])
        mock_get_prices.return_value = {'AAPL': 120.0}
        value = stockSimulator.calculate_current_portfolio_value()
        self.assertEqual(value, 1000 + 2 * 120.0)

    @patch('app.stockSimulator.db_manager.get_portfolio_state')
    def test_can_make_transactions_true(self, mock_get_state):
        # No last transaction date
        mock_get_state.return_value = ([1, 1000, 0, None, None], [])
        self.assertTrue(stockSimulator.can_make_transactions())

    @patch('app.stockSimulator.db_manager.get_portfolio_state')
    def test_can_make_transactions_false(self, mock_get_state):
        # Last transaction was today
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        mock_get_state.return_value = ([1, 1000, 0, today, None], [])
        self.assertFalse(stockSimulator.can_make_transactions())

    @patch('app.stockSimulator.db_manager.get_portfolio_state')
    @patch('app.stockSimulator.calculate_current_portfolio_value')
    def test_get_portfolio_report(self, mock_calc_value, mock_get_state):
        mock_get_state.return_value = ([1, 1000, 0, None, None], [(1, 'AAPL', 2, 100.0, 200.0)])
        mock_calc_value.return_value = 1200
        report = stockSimulator.get_portfolio_report()
        self.assertIn("Cash Balance", report)
        self.assertIn("Current Total Portfolio Value", report)

    @patch('app.stockSimulator.db_manager.get_latest_stock_close_price')
    @patch('app.stockSimulator.db_manager.get_transactions_df')
    @patch('app.stockSimulator.yf.download')
    def test_reconstruct_holdings_and_value(self, mock_yf_download, mock_get_tx_df, mock_get_latest_price):
        # Prepare fake transactions
        tx_data = pd.DataFrame([
            {'symbol': 'AAPL', 'transaction_type': 'BUY', 'quantity': 2, 'transaction_date': '2024-06-01'},
            {'symbol': 'AAPL', 'transaction_type': 'BUY', 'quantity': 3, 'transaction_date': '2024-06-02'},
            {'symbol': 'AAPL', 'transaction_type': 'SELL', 'quantity': 1, 'transaction_date': '2024-06-03'},
        ])
        mock_get_tx_df.return_value = tx_data
        mock_get_latest_price.return_value = 150.0  # Mock latest close price for AAPL

        # The yf.download mock is not used anymore, but keep it for completeness
        mock_yf_download.return_value = pd.DataFrame({'Close': [150.0]}, index=[pd.Timestamp('2024-06-03')])

        import app.stockSimulator as stockSimulator
        holdings, value = stockSimulator.reconstruct_holdings_and_value('2024-06-03', tx_data)
        self.assertEqual(holdings['AAPL'], 4)
        self.assertEqual(value, 4 * 150.0)

    @patch('app.stockSimulator.db_manager.get_latest_stock_close_price')
    @patch('app.stockSimulator.get_portfolio_state')
    def test_calculate_current_portfolio_value(self, mock_get_portfolio_state, mock_get_latest_price):
        # Set up mock portfolio and holdings
        mock_get_portfolio_state.return_value = (
            [1, 1000, 0, None, None],  # portfolio: id, cash_balance, ...
            [(1, 'AAPL', 2, 100.0, 200.0)]  # holdings: stock_id, symbol, quantity, avg_cost, total_cost
        )
        mock_get_latest_price.return_value = 120.0  # Mock latest close price for AAPL

        import app.stockSimulator as stockSimulator
        value = stockSimulator.calculate_current_portfolio_value()
        # cash_balance + 2 * 120.0 = 1000 + 240 = 1240
        self.assertEqual(value, 1240.0)

if __name__ == '__main__':
    unittest.main()