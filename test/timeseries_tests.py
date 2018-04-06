import unittest
import logging
import os

from forecast.timeseries import TimeSeries


class TimeSeriesTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def tearDown(self):
        pass

    def testForecast(self):
        ts = TimeSeries()
        input_file_path = os.path.join(self.base_dir, 'Account_Historical_Revenue.csv')
        r = ts.forecast(input_file_path)
        logging.info(r)
