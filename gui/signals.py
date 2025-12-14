from PyQt5 import QtCore

class WorkerSignals(QtCore.QObject):
    log = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    train_progress = QtCore.pyqtSignal(int, int, str, str)