import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
import logging

from CleanCode.lmn_tests.change_to_lmn import change_coordinates_to_lmn

logger = logging.getLogger(__name__)
MU_0 = 4e-7 * np.pi
K_B = 1.38e-23
PROTON_MASS = 1.67e-27


def b_largest_in_l_direction(b1_L: float, b2_L: float, b1_M: float, b2_M: float) -> bool:
    """
    Check that the magnitude or the amplitude of the magnetic field in the L direction is larger than in the M direction
    :param b1_L: b on one side of the event for L
    :param b2_L: b on the other side of the event for L
    :param b1_M: b on one side of the event for M
    :param b2_M: b on the other side of the event for M
    :return:
    """
    amplitude_change_l, amplitude_change_m = np.abs(b1_L - b2_L), np.abs(b1_M - b2_M)
    magnitude_change_l, magnitude_change_m = b1_L ** 2 + b2_L ** 2, b1_M ** 2 + b2_M ** 2
    if amplitude_change_l > 1 + amplitude_change_m or magnitude_change_l > 1 + magnitude_change_m:
        return True  # we do not want too close results, in which case it is not a reconnection
    else:
        return False


def multiple_tests(b1: np.ndarray, b2: np.ndarray, v1: np.ndarray, v2: np.ndarray, imported_data, event_date: datetime,
                   L: np.ndarray) -> bool:
    """
    Need two of the following:
        sign of Bl changes before and after the event
        Bn is close to zero
        changes in vl larger than changes in vm or vn
        correlations between vl and Bl change before and after the event
    :param b1: magnetic field before the exhaust
    :param b2: magnetic field after the exhaust
    :param v1: velocity before the exhaust
    :param v2: velocity after the exhaust
    :param imported_data: data to test on
    :param event_date: possible event date
    :param L: L vector
    :return:
    """
    reconnection_points = 0

    if np.sign(b1[0]) != np.sign(b2[0]):  # BL changes sign before and after the exhaust
        reconnection_points += 1
    else:
        logger.debug('sign error')
        return False

    if np.abs(b1[2]) < 10e-15 and np.abs(b2[2]) < 10e-15:  # bn is small and nearly constant
        reconnection_points = reconnection_points + 1
    else:
        logger.debug('bn too big')

    delta_v = np.abs(v1 - v2)
    if delta_v[0] > delta_v[1] and delta_v[0] > delta_v[2]:  # changes in vm and vn are small compared to changes in vl
        reconnection_points = reconnection_points + 1
    else:
        logger.debug('v wrong')

    # changes in bl and vl are correlated on one side and anti-correlated on the other side
    change_coordinates_to_lmn(imported_data, L=L)
    bL_diff = imported_data.data['Bl'].diff() / imported_data.data['Bl'].index.to_series().diff().dt.total_seconds()
    vL_diff = imported_data.data['v_l'].diff() / imported_data.data['v_l'].index.to_series().diff().dt.total_seconds()
    left_correlation = bL_diff.loc[
                       event_date - timedelta(minutes=15): event_date - timedelta(minutes=2)].values * vL_diff.loc[
                       event_date - timedelta(minutes=15): event_date - timedelta(minutes=2)].values
    right_correlation = bL_diff.loc[
                        event_date + timedelta(minutes=2):event_date + timedelta(minutes=15)].values * vL_diff.loc[
                        event_date + timedelta(minutes=2):event_date + timedelta(minutes=15)].values

    if np.sign(np.mean(left_correlation)) != np.sign(np.mean(right_correlation)):
        reconnection_points = reconnection_points + 1
    else:
        logger.debug('correlation error')
    if reconnection_points > 1:
        return True
    else:
        return False


def alfven_speed(rho_1: np.ndarray, b1_L: float, v1_L: float, rho_2: np.ndarray, b2_L: float) -> Tuple[float, float]:
    """
    Finds the theoretical speed after the exhaust
    :param rho_1: density before the exhaust
    :param b1_L: magnetic field in the L direction before the exhaust
    :param v1_L: velocity before the exhaust
    :param rho_2: density after the exhaust
    :param b2_L: magnetic field after the exhaust
    :return:
    """
    rho_1 = rho_1 * PROTON_MASS / 1e-15  # density is in cm-3, we want in km-3
    rho_2 = rho_2 * PROTON_MASS / 1e-15  # density is in cm-3, we want in km-3
    alpha_1 = 0  # rho_1 * k * (T_par_1 - T_perp_1) / 2
    alpha_2 = 0  # rho_2 * k * (T_par_2 - T_perp_2) / 2
    b1_part = b1_L * np.sqrt((1 - alpha_1) / (MU_0 * rho_1)) * 10e-10  # b is in nT
    b2_part = b2_L * np.sqrt((1 - alpha_2) / (MU_0 * rho_2)) * 10e-10  # b is in nT
    theoretical_v2_plus = v1_L + (b2_part - b1_part)
    theoretical_v2_minus = v1_L - (b2_part - b1_part)
    return theoretical_v2_plus, theoretical_v2_minus


def walen_test(b1_L: float, b2_L: float, v1_L: float, v2_L: float, rho_1: np.ndarray, rho_2: np.ndarray,
               minimum_fraction: float = 0.9, maximum_fraction: float = 1.1) -> bool:
    """
    Applies the Walen test on the data
    :param b1_L: magnetic field in the L direction before thee exhaust
    :param b2_L: magnetic field in the L direction after the exhaust
    :param v1_L: velocity in the L direction before the exhaust
    :param v2_L: velocity in the L direction after the exhaust
    :param rho_1: density before the exhaust
    :param rho_2: density after the exhaust
    :param minimum_fraction: minimum walen fraction
    :param maximum_fraction: maximum walen fraction
    :return:
    """
    theoretical_v2_plus, theoretical_v2_minus = alfven_speed(rho_1, b1_L, v1_L, rho_2, b2_L)
    logger.debug(theoretical_v2_plus, theoretical_v2_minus, v2_L)
    # the true v2 must be close to the predicted one, we will take the ones with same sign for comparison
    # if they all have the same sign, we compare to both v_plus and v_minus
    if np.sign(v2_L) == np.sign(theoretical_v2_plus) and np.sign(v2_L) == np.sign(theoretical_v2_minus):
        theoretical_v2_min = np.min([np.abs(theoretical_v2_minus), np.abs(theoretical_v2_plus)])
        theoretical_v2_max = np.max([np.abs(theoretical_v2_minus), np.abs(theoretical_v2_plus)])
        if minimum_fraction * np.abs(theoretical_v2_min) < np.abs(v2_L) < maximum_fraction * np.abs(theoretical_v2_max):
            return True
    elif np.sign(v2_L) == np.sign(theoretical_v2_plus):
        if minimum_fraction * np.abs(theoretical_v2_plus) < np.abs(v2_L) < maximum_fraction * np.abs(
                theoretical_v2_plus):
            return True
    elif np.sign(v2_L) == np.sign(theoretical_v2_minus):
        if minimum_fraction * np.abs(theoretical_v2_minus) < np.abs(v2_L) < maximum_fraction * np.abs(
                theoretical_v2_minus):
            return True
    else:
        logger.debug('wrong result')
    return False
