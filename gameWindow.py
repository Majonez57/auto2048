from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QShortcut, QMessageBox
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("2048 Game")

        central_widget = QWidget()
        grid_layout = QGridLayout()
        central_widget.setLayout(grid_layout)

        self.labels = []
        for i in range(4):
            row = []
            for j in range(4):
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; }")
                grid_layout.addWidget(label, i, j)
                row.append(label)
            self.labels.append(row)

        self.score_label = QLabel(f"Score: {0}")
        self.score_label.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; }")
        grid_layout.addWidget(self.score_label, 4, 0, 1, 4)

        self.fitness_label = QLabel(f"Fitness: {0}")
        self.fitness_label.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; }")
        grid_layout.addWidget(self.fitness_label, 5, 0, 1, 4)

        self.setCentralWidget(central_widget)

        self.show()
        self.setFocus()

        self.closed = False

    def update_board(self, board, score, fitness):
        self.game_board = board

        for i in range(4):
            for j in range(4):
                value = board[i][j]
                if value == 0:
                    self.labels[i][j].setText("")
                    self.labels[i][j].setStyleSheet("QLabel { font-size: 24px; font-weight: bold; background-color: #cdc1b4; padding: 10px; }")
                else:
                    self.labels[i][j].setText(str(value))
                    color = self.get_color_for_value(value)
                    self.labels[i][j].setStyleSheet(f"QLabel {{ font-size: 24px; font-weight: bold; background-color: {color}; padding: 10px; }}")

        self.score_label.setText(f"Score: {score}")
        self.fitness_label.setText(f"Fitness: {fitness}")

        self.show()

    def get_color_for_value(self, value):
        colors = {
            2: "#eee4da",
            4: "#ede0c8",
            8: "#f2b179",
            16: "#f59563",
            32: "#f67c5f",
            64: "#f65e3b",
            128: "#edcf72",
            256: "#edcc61",
            512: "#edc850",
            1024: "#edc53f",
            2048: "#edc22e",
        }
        if value in colors:
            return colors[value]
        else:
            return "#cdc1b4"  # Default color for higher values

    def show_game_over_message(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Game Over")
        msg_box.setText(f"Game over! Your final score is {self.game_board.score}.")
        msg_box.exec_()

class Renderer:
    def __init__(self, h=100, w=100, ownWindow = False):
        self.width = w
        self.height = h

        self.window = None

        if ownWindow:
            self.app = QApplication([])
            self.window = GameWindow()
        
        def close():
            pass

        def updateFrame(self):
            pass



