import numpy as np
import pandas as pd


def create_vp_magnitude_column(data: pd.DataFrame)-> pd.DataFrame:
    data['vp_magnitude'] = np.sqrt((data['vp_x']) ** 2 + (data['vp_y']) ** 2 + (data['vp_z']) ** 2)
    return data


def create_b_magnitude_column(data: pd.DataFrame) -> pd.DataFrame:
    data['b_magnitude'] = np.sqrt((data['Bx']) ** 2 + (data['By']) ** 2 + (data['Bz']) ** 2)
    return data


SUPPORTED_COLUMNS = {
    'vp_magnitude': create_vp_magnitude_column,
    'b_magnitude': create_b_magnitude_column,
}
