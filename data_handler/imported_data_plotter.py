from typing import List
import matplotlib.pyplot as plt
from data_handler.imported_data import ImportedData

DEFAULT_PLOTTED_COLUMNS = ['n_p',
                           ('Tp_perp', 'Tp_par'),
                           ('Bx', 'vp_x'),
                           ('By', 'vp_y'),
                           ('Bz', 'vp_z'),
                           ('b_magnitude', 'vp_magnitude')]


def plot_imported_data(imported_data: ImportedData, columns_to_plot: List[str] = DEFAULT_PLOTTED_COLUMNS, save=False):
    """
    Plots given set of columns for a given ImportedData
    :param imported_data: ImportedData
    :param columns_to_plot: list of column names
    :param save: if True, saves generated plot instead of showing
    :return:
    """
    fig, axs = plt.subplots(len(columns_to_plot), 1, sharex=True, figsize=(15, 8.5))
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    axs[0].set_title('Helios ' + str(imported_data.probe) + ' between ' + str(
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
                    ax = ax.twinx()  # creates new ax which shares x
                    subplot_plot_count += 1
        # ax.legend(loc=1)

    if not save:
        plt.show()
    else:
        fig.set_size_inches((15, 10), forward=False)
        plt.savefig('helios_{}_{:%Y_%m_%d_%H}_interval_{}_hours.png'.format(imported_data.probe,
                                                                            imported_data.start_datetime,
                                                                            imported_data.duration),
                    bbox_inches='tight')
        # plt.savefig('helios' + str(imported_data.probe) + '_' + str(
        #     imported_data.start_datetime.strftime('%Y_%m_%d_%H')) + '_interval_' + str(
        #     imported_data.duration) + '_hours' + '.png',
        #             bbox_inches='tight')


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
    ax.plot(imported_data.data[column_name], '-', markersize=2, label=column_name, color=colour)
    ax.set_ylabel(column_name, color=colour)
    ax.grid()


if __name__ == '__main__':
    plot_imported_data(ImportedData(start_date='09/02/1980', start_hour=0, duration=3, probe=1))
