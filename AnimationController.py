import matplotlib.animation as animation
import matplotlib.pyplot as plt


class ControlledAnimation:
    def __init__(self, figc, animate, frames=100, interval=1, repeat=False):
        self.figc = figc
        self.animate = animate
        self.frames = frames
        self.interval = interval
        self.repeat = repeat
        self.ani = animation.FuncAnimation(self.figc, self.animate, frames=self.frames, interval=self.interval,
                                           repeat=self.repeat)
        self.pause = False

    @staticmethod
    def start():
        # FFwriter = animation.FFMpegWriter(fps=60)
        # self.ani.save('input_follower.mp4', writer=FFwriter)
        plt.show()

    def stop(self):
        self.ani.event_source.stop()

    def pause(self):
        self.pause ^= True
        if not self.pause:
            print("halted")
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()