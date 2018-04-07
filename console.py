import getopt
import logging
import os
import sys
from datetime import datetime

from forecast.timeseries import TimeSeries


def main(argv):
    working_dir = ''
    method = 'Holt-Linear'
    usage = 'timeseries.py -d <working_dir> (optional. default is current directory.)' \
            ' -m <method> (optional. possible value should be "l"(Holt-Linear) or "w"(Holt-Winters). Default is "l")'
    try:
        opts, args = getopt.getopt(argv, "hd:m", ["dir="])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-d", "--working_dir"):
            working_dir = arg
        elif opt in ("-m", "--method"):
            if arg == "w":
                method = "Holt-Winters"

    print('working_dir is "%s". method is %s. "' % (working_dir, method))

    ts = TimeSeries()
    base_dir = ''
    if working_dir is not None:
        base_dir = working_dir

    current_time = datetime.now()
    forecast_file_path = os.path.join(base_dir,
                                      'forecast_%s.csv' % current_time.strftime("%Y%m%d%H%M"))
    adjusted_forecast_file_path = os.path.join(base_dir,
                                               'adjusted_forecast_%s_senario1.csv' % current_time.strftime(
                                                   "%Y%m%d%H%M"))
    print("loading historical data for training.")
    historical_data_df = ts.get_train_data_from_db()
    final_goal_df = ts.get_goal_from_db()

    print("forecast start with %s method." % method)
    r = ts.forecast_revenue(historical_data_df, final_goal_df, forecast_file_path, adjusted_forecast_file_path,
                            set_negative_to_zero=True, method=method)
    logging.info(r)

    print("forecast finished.")
    print('output the forecast file to %s' % forecast_file_path)
    print('output the adjusted forecast file to %s' % adjusted_forecast_file_path)


if __name__ == '__main__':
    main(sys.argv[1:])