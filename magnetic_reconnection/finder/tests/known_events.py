from datetime import datetime, timedelta
import json
from typing import List

from magnetic_reconnection.magnetic_reconnection import MagneticReconnection


def get_known_magnetic_reconnections() -> List[MagneticReconnection]:
    with open('known_events.json', 'r') as f:
        loaded_json = json.load(f)

    def parse_json_dict_to_magnetic_reconnection(json_dict):
        start_datetime = datetime(year=json_dict['year'],
                                  month=json_dict['month'],
                                  day=json_dict['day'],
                                  hour=int(json_dict['start_time'][:2]),
                                  minute=int(json_dict['start_time'][-2:]))
        end_datetime = datetime(year=json_dict['year'],
                                month=json_dict['month'],
                                day=json_dict['day'],
                                hour=int(json_dict['end_time'][:2]),
                                minute=int(json_dict['end_time'][-2:]))
        duration: timedelta = end_datetime - start_datetime
        return MagneticReconnection(start_datetime=start_datetime, duration=duration, probe=json_dict['probe'])

    magnetic_reconnections = [parse_json_dict_to_magnetic_reconnection(json_dict) for json_dict in loaded_json]
    # list(map(parse_json_dict_to_magnetic_reconnection, loaded_json))
    return magnetic_reconnections


if __name__ == '__main__':
    import pprint
    pprint.pprint(get_known_magnetic_reconnections())
