import logging
from datetime import datetime


def timer(cls):
    def _timer(func):
        def __timer(*args, **kwargs):
            start_time = datetime.now()
            logging.debug("Start to call method '%s' at %s." % (func.__name__, start_time))
            try:
                func(*args, **kwargs)
            finally:
                stop_time = datetime.now()
                logging.debug("Stop to call method '%s' at %s." % (func.__name__, stop_time))
                logging.info("It takes %s to finish the call of method '%s'." %
                             (str(stop_time - start_time), func.__name__))

        return __timer

    return _timer
