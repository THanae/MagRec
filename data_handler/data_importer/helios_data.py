from heliopy.data import helios

from data_handler.data_importer.imported_data import ImportedData


class HeliosData(ImportedData):
    def __init__(self, start_date: str = '27/01/1976', duration: int = 15, start_hour: int = 0, probe: int = 2):
        """
        :param start_date: string of 'DD/MM/YYYY'
        :param duration: int in hours
        :param start_hour: int from 0 to 23 indicating starting hour of given start_date
        :param probe: 1 for Helios 1, 2 for Helios 2
        """
        super().__init__(start_date, duration, start_hour, probe)
        self.data = self.get_imported_data()
        if len(self.data) == 0:
            raise RuntimeWarning('Created ImportedData object has retrieved no data: {}'.format(self))

    def __repr__(self):
        return '{}: at {:%H:%M %d/%m/%Y} by probe {}. Data has {} entries.'.format(self.__class__.__name__,
                                                                                   self.start_datetime,
                                                                                   self.probe,
                                                                                   len(self.data))

    def get_imported_data(self):
        data = helios.corefit(self.probe, self.start_datetime, self.end_datetime)
        return data


if __name__ == '__main__':
    x = HeliosData()
    print(x.data)
