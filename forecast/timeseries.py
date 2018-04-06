from datetime import datetime, timedelta
from math import sqrt

import pandas as pd
import logging
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

from db_helper.mysql_helper import MySqlHelper

DEBUG = False
N = 4000


class TimeSeries:
    def __init__(self):
        pass

    def read_data(self, filename):
        df = pd.read_csv(filename)
        return df

    def process_data_into_database(self, filename):
        raw_df = pd.read_csv(filename)
        mysql_helper = MySqlHelper()

        for index, row in raw_df.iterrows():
            region = row['Region']
            master_account_id = row['Master_Account_Id']
            account_name = row['Account_Name']
            industry = row['Industry']
            account_size = row['Account_Size']
            product_name = row['Product']
            bu_name = row['BU']

            col_index = 7
            for c in row[col_index:]:
                # handle the date value
                date = raw_df.columns[col_index].replace('Rev_', '')
                date = pd.to_datetime(date, format='%Y_%m')
                date = date.strftime('%Y-%m-%d')
                insert_sql = '''INSERT INTO `revenue`.`revenue`
                                (`Master_Account_Id`,
                                `product_id`,
                                `bu_region_id`,
                                `revenue`,
                                `date`)
                                
                                SELECT 
                                    %s, p.product_id, br.bu_region_id, %s, %s
                                FROM
                                    product p, bu, region r, bu_region br
                                WHERE
                                    p.Product_Name = %s
                                    AND bu.bu_name = %s
                                    AND r.region_name = %s
                                    AND bu.bu_id = br.bu_id
                                    AND r.region_id = br.region_id;'''
                col_index += 1
                mysql_helper.execute(insert_sql, (master_account_id, c, date, product_name, bu_name, region))


def forecast_revenue(self, input_file_name, final_goal_file_name, method, historical_data_df):
    logging.info("1. Get the historical data.")
    df = self.read_data(input_file_name)
    indexes = ['Region', 'Master_Account_Id', 'Product']
    date_columns = df.columns[7:31].tolist()
    date_columns = [col.replace('Rev_', '') for col in date_columns]
    df.columns = df.columns[0:7].tolist() + date_columns
    train_cols = df.columns[7:31].tolist()
    train_cols = indexes + train_cols
    train = df[train_cols]
    train.index = train[indexes]
    transposed_train = train.transpose()[3:]
    transposed_train.index = pd.to_datetime(transposed_train.index, format='%Y_%m')

    logging.info("2. Forecast based on historical data.")
    forecast_period = 12
    index = pd.date_range('2018-01-01', period=forecast_period, freq='MS')
    forecast_df = pd.DataFrame(index=index, columns=transposed_train.columns)

    i = 0
    for col in transposed_train.columns:
        i += 1
        logging.info("\n\n%s forecast the revenue for customer %d. %s\n\n" % ('*' * 10, i, '*' * 10))
        if DEBUG:
            if i > N:
                break
        train_records = transposed_train.loc[:, [col]]
        col_name = '_'.join(map(str, col))

        logging.info("\nForecasting %s." % col_name)

        if DEBUG:
            # check the trend, seasonality
            sm.tsa.seasonal_decompose(train_records[col]).plot()
            result = sm.tsa.stattools.adfuller(train_records[col])
            plt.savefig("%s_seasonal_decompose.png" % col_name)
            plt.close()

        logging.info("Forecast with %s method." % method)
        if method == 'Holt-Linear':
            fit1 = Holt(np.asarray(train_records[col])).fit(smoothing_level=0.3, smoothing_slope=0.1)
            forecast_df[col] = fit1.forecast(forecast_period)
        elif method == 'Holt-Winters':
            fit2 = ExponentialSmoothing(np.asarray(train_records[col]), seasonal_periods=1,
                                        trend='add', seasonal='add', ).fit()
            forecast_df[col] = fit2.forecast(forecast_period)
        else:
            return None

        # draw the forecast picture with train and test data.
        if DEBUG:
            plt.figure(figsize=(16, 8))
            plt.plot(train_records[col], label='Train')
            plt.plot(forecast_period[col], label=method)
            plt.legend(loc='best')
            plt.savefig("%s_forecast.png" % col_name)
            plt.close()

    logging.info("3. Get the final goal for each BU.")
    final_goal_df = pd.read_csv(final_goal_file_name)

    logging.info("4. Adjust the forecast based on the final goal for each BU.")

    logging.info("5. Output the forecast result to a file.")


def forecast(self, filename):
    logging.info('1. reading data.')

    df = self.read_data(filename)

    logging.info("2. clean the headers.")
    indexes = ['Region', 'Master_Account_Id', 'Product']
    date_columns = df.columns[7:31].tolist()
    date_columns = [col.replace('Rev_', '') for col in date_columns]
    df.columns = df.columns[0:7].tolist() + date_columns

    logging.info("3. handle the trainning data.")
    train_cols = df.columns[7:28].tolist()
    train_cols = indexes + train_cols
    train = df[train_cols]
    train.index = train[indexes]
    transposed_train = train.transpose()[3:]
    transposed_train.index = pd.to_datetime(transposed_train.index, format='%Y_%m')

    logging.info("4. handle the test data.")
    test_cols = indexes + df.columns.values[28:31].tolist()
    test = df[test_cols]
    test.index = test[indexes]
    transposed_test = test.transpose()[3:]
    transposed_test.index = pd.to_datetime(transposed_test.index, format='%Y_%m')

    # forecast dataset.
    y_hat_avg = transposed_test.copy()
    foreast_period = len(y_hat_avg)

    logging.info("4. Iterate all the columns of the dataset to forecast each customer's revenue.")

    transposed_train = transposed_train.astype(np.float64)
    transposed_test = transposed_test.astype(np.float64)

    i = 0
    total_error_of_holt_linear = 0
    total_error_of_holt_winter = 0
    for col in transposed_train.columns:
        i += 1
        if i > N:
            break
        logging.info("\n\n%s forecast the revenue for customer %d. %s\n\n" % ('*' * 10, i, '*' * 10))

        train_records = transposed_train.loc[:, [col]]
        test_records = transposed_test.loc[:, [col]]
        col_name = '_'.join(map(str, col))

        logging.info("\nForecasting %s." % col_name)

        if DEBUG:
            # check the trend, seasonality
            sm.tsa.seasonal_decompose(train_records[col]).plot()
            result = sm.tsa.stattools.adfuller(train_records[col])
            plt.savefig("%s_seasonal_decompose.png" % col_name)
            plt.close()

        logging.info("5.1 Forecast with Holt linear method.")
        fit1 = Holt(np.asarray(train_records[col])).fit(smoothing_level=0.3, smoothing_slope=0.1)
        y_hat_avg['Holt_linear'] = fit1.forecast(foreast_period)
        rms1 = sqrt(mean_squared_error(test_records[col], y_hat_avg['Holt_linear']))
        logging.info("Error from Holt linear method is %f" % rms1)
        total_error_of_holt_linear += rms1

        logging.info("5.2 Forecast with Holt Winter method.")
        fit2 = ExponentialSmoothing(np.asarray(train_records[col]), seasonal_periods=7,
                                    trend='add', seasonal='add', ).fit()
        y_hat_avg['Holt_Winter'] = fit2.forecast(len(transposed_test))

        rms2 = sqrt(mean_squared_error(test_records[col], y_hat_avg['Holt_Winter']))
        logging.info("Error from Holt Winter method is %f" % rms2)
        total_error_of_holt_winter += rms2

        # draw the forecast picture with train and test data.
        if DEBUG:
            plt.figure(figsize=(16, 8))
            plt.plot(train_records[col], label='Train')
            plt.plot(test_records[col], label='Test')
            plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
            plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
            plt.legend(loc='best')
            plt.savefig("%s_forecast.png" % col_name)
            plt.close()

    logging.info("The total error from Holt linear method is %f." % total_error_of_holt_linear)
    logging.info("The total error from Holt winter method is %f." % total_error_of_holt_winter)


if __name__ == '__main__':
    filename = ''
    ts = TimeSeries()
    ts.forecast(filename)
