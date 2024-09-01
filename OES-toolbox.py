#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt6.QtWidgets import QSplashScreen, QApplication
from PyQt6.QtGui import QPixmap
from OES_toolbox.ui import resources # seems unused but is needed!
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)

    pixmap = QPixmap(":/images/splash.png")
    splash = QSplashScreen(pixmap)
    splash.show()

    from OES_toolbox import run
    run(app, splash)

