from PySide6.QtWidgets import QApplication, QLabel
import sys


def main():
    app = QApplication(sys.argv)

    label = QLabel("MaiOCR v0.1")
    label.resize(300,100)
    label.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()