import pyaudio
import numpy as np
import time

import cv2

class AudioTrigger:
    def __init__(self, RATE=44100, CHUNK=600): # by default, CHUNKS represent ~15ms of recording
        # set parameters
        self.RATE = RATE
        self.CHUNK = CHUNK
        #cv2.namedWindow('audio', cv2.WINDOW_NORMAL)

    def load(self):
        self.stupid = 0

        # Initialization
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)

    def free(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def check(self, THRESH=32000):
        # Read the Mic adn convert to numpy array
        while self.stream.get_read_available() > 0:
            data = self.stream.read(self.CHUNK)
            data = np.fromstring(data, dtype=np.int16)

            # Test if there is a value above threshold
            # print data.shape, data.std()
            result = np.sum(data > THRESH) > self.CHUNK/2

            # emulate a check
            #self.stupid += 100

            # free stream if trigger was
            if result:
                self.free()
            return result

        return False

if __name__ == "__main__":
    A = AudioTrigger()
    A.load()
    while not A.check(): time.sleep(.01)
