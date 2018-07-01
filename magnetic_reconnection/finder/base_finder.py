from data_handler.imported_data import ImportedData


class BaseFinder:
    """
    Finder objects can be initialised with parameters such as thresholds. Finders can then be run on
    """

    def __init__(self, **kwargs):
        """
        Initialises Finder object with parameters
        :param kwargs: Parameters for Finder initialisation (e.g. thresholds)
        """
        # raise NotImplementedError("Finder classes should have an __init__ that initialises parameters")
        pass

    def find_magnetic_reconnections(self, imported_data: ImportedData):
        raise NotImplementedError("Finder classes should have a find_magnetic_reconnections.")

