from datetime import datetime, timedelta


class MagneticReconnection:
    def __init__(self, start_datetime: datetime, duration: timedelta, probe: int):
        self.start_datetime: datetime = start_datetime
        self.duration: timedelta = duration
        self.probe: int = probe

    def __repr__(self):
        return '{}: at {:%H:%M %d/%m/%Y} by probe {}'.format(self.__class__.__name__,
                                                             self.start_datetime,
                                                             self.probe)
