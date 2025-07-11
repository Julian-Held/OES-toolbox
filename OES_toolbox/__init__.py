#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt6.QtWidgets import QSplashScreen, QApplication
from PyQt6.QtGui import QPixmap
from OES_toolbox.ui import resources # seems unused but is needed!
import sys

from importlib import metadata

from ._version import __version__

def main():
    app = QApplication(sys.argv)

    pixmap = QPixmap(":/images/splash.png")
    splash = QSplashScreen(pixmap)
    splash.show()

    from .toolbox import run
    run(app, splash)

if __name__ == '__main__':
    main()
