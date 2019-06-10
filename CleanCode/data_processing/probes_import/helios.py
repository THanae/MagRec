from heliopy.data import helios
from datetime import datetime, timedelta


def helios_data(start_date: str = '27/01/1976', duration: int = 15, start_hour: int = 0, probe: int = 2):
    start_datetime = datetime.strptime(start_date + '/%i' % start_hour, '%d/%m/%Y/%H')
    end_datetime = start_datetime + timedelta(hours=duration)
    data = helios.corefit(probe, start_datetime, end_datetime)
    return data.data


if __name__ == '__main__':
    # x = HeliosData(start_date='17/01/1976', duration=35040, probe=2)
    x = helios_data(start_date='06/01/1980', duration=3, probe=1)
    print(x)
