#drawNumber

try:
    import sys, os
    stdout = sys.__stdout__
    stderr = sys.__stderr__
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    import pygame
    import tensorflow as tf
    # import matplotlib.pyplot as plt
    import numpy as np
    from tkinter import *
    from tkinter import messagebox
    sys.stdout = stdout
    sys.stderr = stderr
except:
    import sys, os
    stdout = sys.__stdout__
    stderr = sys.__stderr__
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    import pygame
    import tensorflow as tf
    # import matplotlib.pyplot as plt
    import numpy as np
    from tkinter import *
    from tkinter import messagebox
    sys.stdout = stdout
    sys.stderr = stderr

class Pixel(object):

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255,255,255)
        self.neighbours = []

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.x + self.width, self.y + self.height))
    
    def getNeighbour(self, g):
        j = self.x // 20
        i = self.y // 20
        rows = 28
        cols = 28

        #hor_ and ver_ neighbours
        if i < cols - 1:
            self.neighbours.append(g.pixels[i+1][j])
        if i > 0:
            self.neighbours.append(g.pixels[i-1][j])
        if j < rows - 1:
            self.neighbours.append(g.pixels[i][j+1])
        if j > 0:
            self.neighbours.append(g.pixels[i][j-1])
        
        #diognal neighbours
        if j > 0 and i > 0:
            self.neighbours.append(g.pixels[i-1][j-1])
        if (j+1 < rows) and (i > -1) and i > 0:
            self.neighbours.append(g.pixels[i-1][j+1])
        if (j-1 < rows) and (i < cols -1) and (j-1 >0):
            self.neighbours.append(g.pixels[i+1][j-1])
        if (j < rows-1) and (i < cols - 1):
            self.neighbours.append(g.pixels[i+1][j+1])



class Grid(object):
    pixels = []

    def __init__(self, row, col, width, height):
        self.rows = row
        self.cols = col
        self.len = row*col
        self.width = width
        self.height = height
        self.generatePixels()
        pass
    
    def draw(self, surface):
        for row in self.pixels:
            for col in row:
                col.draw(surface)
    
    def generatePixels(self):
        x_gap = self.width // self.cols
        y_gap = self.height // self.rows
        self.pixels = []
        for r in range(self.rows):
            self.pixels.append([])
            for c in range(self.cols):
                self.pixels[r].append(Pixel(x_gap * c, y_gap * r, x_gap, y_gap))
        
        for r in range(self.rows):
            for c in range(self.cols):
                self.pixels[r][c].getNeighbour(self)
    
    def clicked(self, pos):
        try:
            t = pos[0]
            w = pos[1]
            g1 = int(t) // self.pixels[0][0].width
            g2 = int(w) // self.pixels[0][0].height
            return self.pixels[g2][g1]
        except:
            pass
    
    def convert_binary(self):
        li = self.pixels
        matrix = [[] for x in range(len(li))]

        for i in range(len(li)):
            for j in range(len(li)):
                if li[i][j].color == (255,255,255):
                    matrix[i].append(0)
                else:
                    matrix[i].append(1)
        
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_test = tf.keras.utils.normalize(x_test, axis=1)
        for row in range(28):
            for x in range(28):
                x_test[0][row][x] = matrix[row][x]
        return x_test[:1]


def guess(li, model):
    predictions = model.predict(li)
    print(predictions[0])
    t = (np.argmax(predictions[0]))
    print("I predict this number is : ", t)
    window = Tk()
    window.withdraw()
    messagebox.showinfo("Prediciton","I predict this number is: "+str(t))
    window.destroy()



def main():
    run = True
    model = tf.keras.models.load_model('num_reader.model')
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                li = g.convert_binary()
                guess(li,model)
                g.generatePixels()
            
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                clicked = g.clicked(pos)
                clicked.color = (0,0,0)
                for n in clicked.neighbours:
                    n.color = (0,0,0)
            
            if pygame.mouse.get_pressed()[2]:
                try:
                    pos = pygame.mouse.get_pos()
                    clicked = g.clicked(pos)
                    clicked.color = (255,255,255)
                except:
                    pass
        
        g.draw(win)
        pygame.display.update()
            

pygame.init()
width = height = 560
win = pygame.display.set_mode((width,height))
pygame.display.set_caption("Number Guesser")
g = Grid(28,28,width, height)
main()

pygame.quit()
quit()
