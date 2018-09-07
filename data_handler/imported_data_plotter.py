from datetime import datetime
from typing import List, Optional, Union, Tuple
import matplotlib.pyplot as plt

from data_handler.data_importer.data_import import get_probe_data
from data_handler.data_importer.imported_data import ImportedData

DEFAULT_PLOTTED_COLUMNS = ['n_p',
                           ('Tp_perp', 'Tp_par'),
                           ('Bx', 'vp_x'),
                           ('By', 'vp_y'),
                           ('Bz', 'vp_z'),
                           ('b_magnitude', 'vp_magnitude')]


def plot_imported_data(imported_data: ImportedData,
                       columns_to_plot: List[Union[str, Tuple[str, str]]] = DEFAULT_PLOTTED_COLUMNS, save=False,
                       event_date: Optional[datetime] = None, boundaries: Optional[List[datetime]] = None,
                       scatter_points: Optional[list] = None):
    """
    Plots given set of columns for a given ImportedData
    :param imported_data: ImportedData
    :param columns_to_plot: list of column names
    :param save: if True, saves generated plot instead of showing
    :param event_date: date of event to be marked on plot, None if no event to be indicated
    :param boundaries: boundaries of the event to be indicated on plot, None if no boundaries to be indicated
    :param scatter_points: points to be scattered on the plots
    :return:
    """
    fig, axs = plt.subplots(len(columns_to_plot), 1, sharex='all', figsize=(15, 8.5))
    # colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colours = ['m', 'b'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    axs[0].set_title('Probe ' + str(imported_data.probe) + ' between ' + str(
        imported_data.start_datetime.strftime('%d/%m/%Y')) + ' and ' + str(
        imported_data.end_datetime.strftime('%d/%m/%Y')) + ' at ' + str(
        imported_data.data['r_sun'].values[0]) + ' astronomical units')
    for ax_index in range(len(columns_to_plot)):
        subplot_plot_count = 0
        ax = axs[ax_index]
        if isinstance(columns_to_plot[ax_index], str):
            column_to_plot = columns_to_plot[ax_index]
            plot_to_ax(imported_data, ax=ax, column_name=column_to_plot, colour=colours[subplot_plot_count])
        else:
            assert len(columns_to_plot[ax_index]) == 2, 'Can only create 2 plots per subplot, not %s. Thank you.' % len(
                columns_to_plot[ax_index])
            for column_to_plot in columns_to_plot[ax_index]:
                plot_to_ax(imported_data, ax=ax, column_name=column_to_plot, colour=colours[subplot_plot_count])

                if subplot_plot_count == 0:
                    if column_to_plot != 'Tp_perp' and column_to_plot != 'Tp_par':
                        ax = ax.twinx()  # creates new ax which shares x
                    else:
                        ax = fig.add_subplot(int(str(len(columns_to_plot)) + str(1) + str(ax_index + 1)), sharey=ax,
                                             frameon=False)
                        ax.xaxis.set_ticklabels([])
                    subplot_plot_count += 1

                if scatter_points is not None:
                    for scatter_point in scatter_points:
                        if scatter_point[0] == column_to_plot:
                            ax.scatter(scatter_point[1], scatter_point[2])

        # ax.legend(loc=1)
        if event_date is not None:
            ax.axvline(x=event_date, linewidth=1.5, color='k')
        if boundaries is not None:
            for n in range(len(boundaries)):
                ax.axvline(x=boundaries[n], linewidth=1.2, color='k')

    if not save:
        plt.show()
    else:
        fig.set_size_inches((15, 10), forward=False)
        if event_date is None:
            plt.savefig('helios_{}_{:%Y_%m_%d_%H}_interval_{}_hours.png'.format(imported_data.probe,
                                                                                imported_data.start_datetime,
                                                                                imported_data.duration),
                        bbox_inches='tight')
        else:
            plt.savefig('h{}_{:%Y_%m_%d_%H_%M}_all.png'.format(imported_data.probe, event_date))


def plot_to_ax(imported_data: ImportedData, ax, column_name: str, colour='b'):
    """
    Plots given column of given ImportedData to a given ax.
    :param imported_data: ImportedData
    :param ax: matplotlib ax
    :param column_name: str
    :param colour: matplotlib color
    :return:
    """
    if column_name not in imported_data.data.columns.values:
        imported_data.create_processed_column(column_name)
    ax.plot(imported_data.data[column_name], 'o-', markersize=2, label=column_name, color=colour)
    if column_name == 'Tp_par':
        ax.yaxis.set_label_position("right")
    ax.set_ylabel(column_name, color=colour)
    ax.grid()


if __name__ == '__main__':
    data = get_probe_data(probe='ace', start_date='01/01/2002', start_hour=12, duration=15)
    # data = get_probe_data(probe=1, start_date='09/02/1980', start_hour=0, duration=3)
    # data = get_probe_data(probe=1, start_date='29/05/1981', start_hour=12, duration=6)
    # data = get_probe_data(probe='ulysses', start_date='09/02/1998', duration=24)
    # data = get_probe_data(probe='ulysses', start_date='15/02/2003', start_hour=20, duration=6)

    plot_imported_data(data)
