import pygame as pyg
from PIL import Image

# initial variables
drawing = False 
last_pos = None
width = 9 # width of the line we draw
pen_color = (255, 255, 255) # line is white, background black- model trained on white text on black background

# method for creating the games screen + returning the image
def start_drawing():
    global screen
    pyg.display.set_caption("Hand Drawing Prediction")
    pyg.init() # initialize the window
    screen = pyg.display.set_mode((400, 400)) # image and screen is 400x400
    img = window_loop() # loop until we get an image back
    return img # return the image

def draw(event):
    global drawing, last_pos, width
 
    if event.type == pyg.MOUSEMOTION:
        if (drawing):
            mouse_pos = pyg.mouse.get_pos()
            if last_pos is not None:
                pyg.draw.line(screen, pen_color, last_pos, mouse_pos, width)
            last_pos = mouse_pos
    elif event.type == pyg.MOUSEBUTTONUP:
        drawing = False
        last_pos = None
    elif event.type == pyg.MOUSEBUTTONDOWN:
        drawing = True
 
 
def window_loop():
    global screen
 
    play = 1
    while play:
        for event in pyg.event.get():
            if event.type == pyg.KEYDOWN:
                if event.key == pyg.K_s:
                    image_bytes = pyg.image.tostring(screen, "RGB")
                    image_bytes2 = Image.frombytes("RGB", (400,400), image_bytes)
                    return image_bytes2
            if event.type == pyg.QUIT:
                play = 0
            draw(event)
        pyg.display.update()
    pyg.display.quit()

