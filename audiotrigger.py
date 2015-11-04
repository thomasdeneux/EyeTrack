import pyaudio
import numpy as np
import time

class AudioTrigger:
    def __init__(self, RATE=44100, BUFFER=1000):
        # set parameters
        self.RATE = RATE
        self.BUFFER = BUFFER

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
                                  frames_per_buffer=self.BUFFER)  # buffer 200ms

    def __del__(self):
        self.free()

    def free(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def check(self, THRESH=10000):
        # Read the Mic adn convert to numpy array
        data = self.stream.read(self.stream.get_read_available())
        data = np.fromstring(data, dtype=np.int16)

        # Test if there is a value above threshold
        #print data.mean()
        result = np.sum(data > THRESH) > 0

        # emulate a check
        self.stupid += 100
        result = (self.stupid >= THRESH)

        # free stream if trigger was detected
        if result:
            self.free()
        return result


if __name__ == "__main__":
    A = AudioTrigger()
    A.load()
    time.sleep(.01)
    print A.check()
    time.sleep(.1)
    print A.check()
    time.sleep(.2)
    print A.check()
    time.sleep(.5)
    print A.check()
    time.sleep(1)
    print A.check()
    time.sleep(2)
    print A.check()
    time.sleep(5)
    print A.check()
