from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Ellipse, Color, Line
import tensorflow as tf
import numpy as np
import cv2
import os

Window.size = (560, 610)
Window.clearcolor = (1, 1, 1, 1)
model = tf.keras.models.load_model('num_reader.model')


class MyLayout(BoxLayout):

    def export(self, *args):
        self.ids.myexport.export_to_png("test2.png")


class PaintWindow(Widget):

    def on_touch_down(self, touch):
        self.canvas.add(Color(rgb=(0, 0, 0)))
        d = 50
        self.canvas.add(Ellipse(pos=(touch.x - d, touch.y - d), size=(2 * d, 2 * d)))
        self.points = Line(pos=(touch.x, touch.y), width=d)
        touch.ud['line'] = self.points
        self.canvas.add(touch.ud['line'])

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class NumGuessApp(App):

    def build(self):
        self.rootWindow = MyLayout(size=(500, 300))
        self.painter = PaintWindow()
        self.guessBtn = Button(text='Guess', size_hint=(None, None),
                               pos_hint={'center_x': 1}, height=50, width=Window.size[0])
        self.guessBtn.bind(on_release=self.guess)
        self.rootWindow.add_widget(self.painter)
        self.rootWindow.add_widget(self.guessBtn)

        return self.rootWindow

    def guess(self, obj):
        img_name = Window.screenshot()
        img = cv2.imread(img_name, 0)[:-50, :]
        os.remove(img_name)
        guessNum(img)
        self.painter.canvas.clear()


def guessNum(img):
    img = (255 - img)
    img = img // 255
    img = cv2.resize(img, dsize=(28, 28))
    img = img.astype(int)
    img = img // 1.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict([img])
    guess = np.argmax(predictions[0])
    print("I predict this number is a:", guess)


NumGuessApp().run()
