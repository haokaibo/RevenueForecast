import unittest
import logging
import os
from datetime import datetime
import pandas as pd

from forecast.timeseries import TimeSeries


class TimeSeriesTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def tearDown(self):
        pass

    def test_forecast_test(self):
        ts = TimeSeries()
        input_file_path = os.path.join(self.base_dir, 'Account_Historical_Revenue.csv')
        r = ts.forecast_test(input_file_path)
        logging.info(r)

    def test_process_data_into_database(self):
        ts = TimeSeries()
        input_file_path = os.path.join(self.base_dir, 'Account_Historical_Revenue.csv')
        r = ts.process_data_into_db(input_file_path)
        logging.info(r)

    def test_get_train_data_from_db(self):
        ts = TimeSeries()
        r = ts.get_train_data_from_db()
        logging.info(r)

    def test_get_goal_from_db(self):
        ts = TimeSeries()
        r = ts.get_goal_from_db()
        logging.info(r)

    def test_forecast_revenue_by_Holt_linear_not_setting_negtive_forecast_to_zero(self):
        scenario = "scenario1"
        ts = TimeSeries()

        current_time = datetime.now()
        forecast_file_path = os.path.join(self.base_dir,
                                          'forecast_%s_%s.csv' % (current_time.strftime("%Y%m%d%H%M"), scenario))
        adjusted_forecast_file_path = os.path.join(self.base_dir,
                                                   'adjusted_forecast_%s_%s.csv' % (current_time.strftime(
                                                       "%Y%m%d%H%M"), scenario))
        historical_data_df = ts.get_train_data_from_db()
        final_goal_df = ts.get_goal_from_db()
        r = ts.forecast_revenue(historical_data_df, final_goal_df, forecast_file_path, adjusted_forecast_file_path,
                                set_negative_to_zero=False)
        logging.info(r)

    def test_forecast_revenue_by_Holt_linear_setting_negtive_forecast_to_zero(self):
        scenario = "scenario2"
        ts = TimeSeries()

        current_time = datetime.now()
        forecast_file_path = os.path.join(self.base_dir,
                                          'forecast_%s_%s.csv' % (current_time.strftime("%Y%m%d%H%M"), scenario))
        adjusted_forecast_file_path = os.path.join(self.base_dir,
                                                   'adjusted_forecast_%s_%s.csv' % (current_time.strftime(
                                                       "%Y%m%d%H%M"), scenario))
        historical_data_df = ts.get_train_data_from_db()
        final_goal_df = ts.get_goal_from_db()
        r = ts.forecast_revenue(historical_data_df, final_goal_df, forecast_file_path, adjusted_forecast_file_path)
        logging.info(r)

    def test_forecast_revenue_by_Holt_winters_not_setting_negtive_forecast_to_zero(self):
        scenario = "scenario3"
        ts = TimeSeries()
        current_time = datetime.now()
        forecast_file_path = os.path.join(self.base_dir,
                                          'forecast_%s_%s.csv' % (current_time.strftime("%Y%m%d%H%M"), scenario))
        adjusted_forecast_file_path = os.path.join(self.base_dir,
                                                   'adjusted_forecast_%s_%s.csv' % (current_time.strftime(
                                                       "%Y%m%d%H%M"), scenario))
        historical_data_df = ts.get_train_data_from_db()
        final_goal_df = ts.get_goal_from_db()
        r = ts.forecast_revenue(historical_data_df, final_goal_df, forecast_file_path, adjusted_forecast_file_path,
                                set_negative_to_zero=False, method='Holt-Winters')
        logging.info(r)

    def test_forecast_revenue_by_Holt_winters_setting_negtive_forecast_to_zero(self):
        scenario = "scenario4"
        ts = TimeSeries()

        current_time = datetime.now()
        forecast_file_path = os.path.join(self.base_dir,
                                          'forecast_%s_%s.csv' % (current_time.strftime("%Y%m%d%H%M"), scenario))
        adjusted_forecast_file_path = os.path.join(self.base_dir,
                                                   'adjusted_forecast_%s_%s.csv' % (current_time.strftime(
                                                       "%Y%m%d%H%M"), scenario))

        historical_data_df = ts.get_train_data_from_db()
        final_goal_df = ts.get_goal_from_db()
        r = ts.forecast_revenue(historical_data_df, final_goal_df, forecast_file_path, adjusted_forecast_file_path,
                                set_negative_to_zero=True, method='Holt-Winters')
        logging.info(r)
