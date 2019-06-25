import cv2
import os
from utils import *
from matplotlib.animation import FuncAnimation

cam = cv2.VideoCapture(0)

# def animate(i):
#     plt.cla()
#     ax = plt.gca()
#     ax.set_xlim(0,100)
#     plt.barh(['spill', 'nospill'], [23,45])
    
# ani = FuncAnimation(plt.gcf(), animate, interval=10)
# plt.tight_layout()
# plt.show()


def predict_cv():
    while True:
        prev_label = ""
        _, img = cam.read()
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        prob, out = predict(im_pil, model)
    #     if str(prev_label) != str(out) and prob > 0.6:
    #             os.system("clear")
    #             print(out)
        cv2.imshow("Image", img)
        # plt.barh(['spill', 'nospill'], [23,45])
        print(f'Probability: {prob}  Class: {out}')
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_cv()
