from util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/pathToDavis16'

    @staticmethod
    def db_root_dir17():
        return '/pathToDavis17'

    @staticmethod
    def save_root_dir():
        return './models'

    @staticmethod
    def models_dir():
        return "./models"

