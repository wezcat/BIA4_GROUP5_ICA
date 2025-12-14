import sys
import torch
from PyQt5 import QtWidgets
from gui.main_window import MainWindow

def main():
    if sys.platform.startswith("win"):
        torch.multiprocessing.set_start_method("spawn", force=True)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()