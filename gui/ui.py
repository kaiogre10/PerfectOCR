import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit
from PySide6.QtCore import QProcess

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PerfectOCR GUI")

        self.layout = QVBoxLayout()

        self.run_button = QPushButton("RUN PIPELINE")
        self.logs = QTextEdit()

        self.logs.setReadOnly(True)

        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.logs)

        self.setLayout(self.layout)

        self.process = QProcess()

        self.run_button.clicked.connect(self.run_pipeline)

        self.process.readyReadStandardOutput.connect(
            self.handle_stdout
        )

        self.process.readyReadStandardError.connect(
            self.handle_stderr
        )

    def run_pipeline(self):
        self.logs.clear()

        self.process.start(
            "python",
            ["main.py"]
        )

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode("utf-8")

        self.logs.append(text)

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        text = bytes(data).decode("utf-8")

        self.logs.append(text)

ui = QApplication(sys.argv)

window = MainWindow()
window.resize(900, 600)
window.show()

sys.exit(ui.exec())