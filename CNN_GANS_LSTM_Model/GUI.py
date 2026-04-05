import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QGroupBox)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np


class TomatoGripperGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TOMATO GRIPPER GUI")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left Panel (30% width)
        left_panel = QGroupBox("📥 LEFT PANEL\nInput & Actions")
        left_layout = QVBoxLayout()

        # Left Panel Buttons
        buttons = [
            "Load Image/Video",
            "Live Camera Feed",
            "Preprocess Input",
            "Classify (Ripe/Unripe/Rotten)",
            "Estimate Grip Force",
            "Run Full Simulation",
            "Export Results (CSV, PDF)"
        ]

        for btn_text in buttons:
            btn = QPushButton(btn_text)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 10px;
                    font-size: 14px;
                    margin: 5px;
                }
            """)
            left_layout.addWidget(btn)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)

        # Right Panel (70% width)
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Visual Data Viewer (Top)
        visual_viewer = QGroupBox("VISUAL DATA VIEWER")
        visual_layout = QVBoxLayout()

        # Matplotlib Figure for camera/image display
        self.fig_camera = Figure(figsize=(10, 6))
        self.canvas_camera = FigureCanvasQTAgg(self.fig_camera)
        self.ax_camera = self.fig_camera.add_subplot(111)
        self.ax_camera.axis('off')  # Hide axes for image display
        self.ax_camera.set_title("Camera/Image Feed")

        # Display sample image (replace with actual camera feed)
        sample_image = np.random.rand(100, 100, 3)
        self.ax_camera.imshow(sample_image)
        self.canvas_camera.draw()

        visual_layout.addWidget(self.canvas_camera)
        visual_viewer.setLayout(visual_layout)

        # Tactile Data Viewer (Bottom Right)
        tactile_viewer = QGroupBox("TACTILE DATA VIEWER")
        tactile_layout = QVBoxLayout()

        # Matplotlib Figure for force heatmap
        self.fig_force = Figure(figsize=(5, 3))
        self.canvas_force = FigureCanvasQTAgg(self.fig_force)
        self.ax_force = self.fig_force.add_subplot(111)

        # Sample force data
        force_data = np.random.rand(10, 10)
        heatmap = self.ax_force.imshow(force_data, cmap='hot')
        self.fig_force.colorbar(heatmap)
        self.ax_force.set_title("Force/Pressure Heatmap")
        self.canvas_force.draw()

        tactile_layout.addWidget(self.canvas_force)
        tactile_viewer.setLayout(tactile_layout)

        # Status Panel (Bottom Left)
        status_panel = QGroupBox("STATUS")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("Tomato Status: Unknown")
        self.force_label = QLabel("Grip Force Estimate: 0.0N")

        for label in [self.status_label, self.force_label]:
            label.setStyleSheet("font-size: 16px; font-weight: bold;")
            status_layout.addWidget(label)

        status_panel.setLayout(status_layout)

        # Bottom Panel (combines status and tactile viewers)
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(status_panel, 50)
        bottom_layout.addWidget(tactile_viewer, 50)
        bottom_panel.setLayout(bottom_layout)

        # Assemble Right Panel
        right_layout.addWidget(visual_viewer, 70)
        right_layout.addWidget(bottom_panel, 30)
        right_panel.setLayout(right_layout)

        # Add panels to main layout
        main_layout.addWidget(left_panel, 30)
        main_layout.addWidget(right_panel, 70)

        # Connect button signals (placeholder functions)
        self.connect_buttons()

    def connect_buttons(self):
        """Connect buttons to their functions"""
        # You'll implement these functions later
        pass


def main():
    app = QApplication(sys.argv)

    # Set dark theme (optional)
    app.setStyle('Fusion')

    window = TomatoGripperGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()