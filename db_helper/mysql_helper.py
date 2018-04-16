from mysql.connector import errorcode

import mysql.connector as conn
import logging

from pandas import DataFrame

from util.timer import timer


class MySqlHelper:
    config = {
        'user': 'root',
        'password': 'mysql_2012',
        'host': '127.0.0.1',
        'database': 'revenue',
        'raise_on_warnings': True,
    }

    @timer(config)
    def query(self, query):

        if query is None or query == '':
            return None
        try:
            con = conn.connect(**MySqlHelper.config)
            r = con.cmd_query(query)

            con.close()
            return r
        except conn.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logging.error("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                logging.error("Database does not exist")
            else:
                logging.error(err)
        else:
            conn.close()

    def execute(self, query, params=None, multi=False):
        if query is None or query == '':
            return None
        try:
            cnx = conn.connect(**MySqlHelper.config)
            cursor = cnx.cursor()

            cursor.execute(query, params=params, multi=multi)

            # Make sure data is committed to the database
            cnx.commit()
            cursor.close()
            cnx.close()
        except conn.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logging.error("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                logging.error("Database does not exist")
            else:
                logging.error(err)
        else:
            cnx.close()

    def fetch_all(self, query, params=None):
        if query is None or query == '':
            return None
        try:
            cnx = conn.connect(**MySqlHelper.config)
            cursor = cnx.cursor()

            iter = cursor.execute(query, params=params)
            df = DataFrame(cursor.fetchall())
            df.columns = cursor.column_names
            # Make sure data is committed to the database
            cnx.commit()
            cursor.close()
            cnx.close()
            return df
        except conn.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logging.error("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                logging.error("Database does not exist")
            else:
                logging.error(err)
        else:
            cnx.close()
            return None