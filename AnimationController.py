import matplotlib.animation as animation
import matplotlib.pyplot as plt


class ControlledAnimation:
    def __init__(self, figc, animate, frames=100, interval=1, video_request=False, repeat=False, progress_bar=False):
        self.figc = figc
        self.animate = animate
        self.frames = frames
        self.interval = interval
        self.repeat = repeat
        self.video_request = video_request
        if not progress_bar:
            self.ani = animation.FuncAnimation(self.figc, self.animate, frames=self.frames, interval=self.interval,
                                               repeat=self.repeat)
        else:
            # noinspection PyTypeChecker
            self.ani = animation.FuncAnimation(self.figc, self.animate, frames=range(self.frames), interval=self.interval,
                                               repeat=self.repeat)
        self.m_pause = False
        self.cid = self.figc.canvas.mpl_connect('button_press_event', self.pause)

    def start(self):
        if self.video_request:
            FFwriter = animation.FFMpegWriter(fps=60)
            self.ani.save('assets/video.mp4', writer=FFwriter)
        else:
            plt.show()

    def stop(self):
        self.ani.event_source.stop()

    def pause(self, event):
        self.m_pause ^= True
        if not self.m_pause:
            print("halted")
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()

    def disconnect(self):
        self.figc.canvas.mpl_disconnect(self.cid)
