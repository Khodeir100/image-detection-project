import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QPushButton, QGraphicsView, QGraphicsScene, QLabel
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage 
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class ImageClassifier(QDialog):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        uic.loadUi(r'D:\Computer vision\project\six categories detection\GUI Detection.ui', self)  # Load your .ui file here

        # Load buttons
        self.load_button_1 = self.findChild(QPushButton, 'pushButton')  # Button for loading first image
        self.load_button_1.clicked.connect(self.load_image)

        self.load_button_2 = self.findChild(QPushButton, 'pushButton_2')  # Button for model predict
        self.load_button_2.clicked.connect(self.predict_with_model)

        self.match_button = self.findChild(QPushButton, 'pushButton_3')  # Button for VGG16 predict
        self.match_button.clicked.connect(self.predict_with_vgg)

        # Image viewer
        self.image_viewer_1 = self.findChild(QGraphicsView, 'graphicsView')  # QGraphicsView for image
        self.scene = QGraphicsScene(self)
        self.image_viewer_1.setScene(self.scene)

        # Result labels
        self.result_label = self.findChild(QLabel, 'label')  # Label to show image class
        self.result_label_2 = self.findChild(QLabel, 'label_2')  # Label to show percentage of class

        # Load models
        self.model = load_model(r'D:\Computer vision\project\six categories detection\model.keras')
        self.vgg_model = load_model(r'D:\Computer vision\project\six categories detection\vgg16.keras')

        # Class names mapping
        self.classes = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

        self.image_path = None

    def load_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg);;All Files (*)", options=options)
        if self.image_path:
            # Load and resize image for viewing
            img = cv2.imread(self.image_path)
            img_resized = cv2.resize(img, (448, 448))  # Resize to 448x448 for display
            height, width, _ = img_resized.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            self.scene.clear()
            self.scene.addPixmap(QPixmap.fromImage(q_img))
            self.image_viewer_1.setScene(self.scene)

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (100, 100))  # Resize to match model input
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    def preprocess_imagev(self, image_path):
        imgv = cv2.imread(image_path)
        imgv = cv2.resize(img, (100, 100))  # Resize to match model input
        imgv = img.astype('float32') / 255.0  # Normalize
        imgv = np.expand_dims(img, axis=0)  # Add batch dimension
        return imgv
    def predict_with_model(self):
        if self.image_path:
            img = self.preprocess_image(self.image_path)
            predictions = self.model.predict(img)
            class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            # Update labels with class names
            self.result_label.setText(f'Class: {self.classes[class_idx]}')
            self.result_label_2.setText(f'Confidence: {confidence:.2f}%')

    def predict_with_vgg(self):
        if self.image_path:
            imgv = self.preprocess_imagev(self.image_path)
            predictions = self.vgg_model.predict(imgv)
            class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            # Update labels with class names
            self.result_label.setText(f'Class: {self.classes[class_idx]}')
            self.result_label_2.setText(f'Confidence: {confidence:.2f}%')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.show()
    sys.exit(app.exec_())