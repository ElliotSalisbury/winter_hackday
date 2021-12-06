from datetime import datetime
import time

import numpy as np
import pyaudio

from tree_animator import TreeAnimator
from utils.color import hsv_to_rgb


class FT():
    # Need to find suitable sample size
    def __init__(self, bit_res=pyaudio.paInt16, channels=1, \
                    sample_rate=44100, sample_size=int(8192/2)):
        # Init parameters
        self.bit_res = bit_res
        self.channels = channels
        self.sample_rate = sample_rate
        self.sample_size = sample_size 
        self.device_index = self.find_mic_idx()
        if self.device_index is None:
            raise Exception("you done messed up") 

        # mic sensitivity correction and bit conversion
        self.mic_sens_dBV = -47.0 # mic sensitivity in dBV + any gain
        self.mic_sens_corr = np.power(10.0, self.mic_sens_dBV/20.0) # calculate mic sensitivity conversion factor

        # create pyaudio instantiation
        audio = pyaudio.PyAudio() 

        # create pyaudio self.stream
        self.stream = audio.open(format = self.bit_res,rate = self.sample_rate,channels = self.channels, \
                            input_device_index = self.device_index,input = True, \
                            frames_per_buffer=self.sample_size)
    
    def find_mic_idx(self):
        ''' Gets device index corresponding to mic port '''
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            print(p.get_device_info_by_index(i))
            dev = p.get_device_info_by_index(i)
            if dev['name'] == 'Microphone (Yeti Nano)' or dev['name'] =='Yeti Nano: USB Audio (hw:3,0)':
                return dev['index']

        return None

    def get_samples(self):
        # record self.samples self.sample_size 
        self.stream.start_stream()
        self.samples = np.fromstring(self.stream.read(self.sample_size),dtype=np.int16)
        self.stream.stop_stream()

        # (USB=5V, so 15 bits are used (the 16th for negatives)) and the manufacturer microphone sensitivity corrections
        self.samples = ((self.samples/np.power(2.0,15))*5.25)*(self.mic_sens_corr) 
        return self.samples

    def get_ft(self):
        # compute FFT parameters
        self.f_vec = self.sample_rate*np.arange(self.sample_size/2)/self.sample_size # frequency vector based on window size and sample rate
        mic_low_freq = 20 # low frequency response of the mic (mine in this case is 20 Hz)
        low_freq_loc = np.argmin(np.abs(self.f_vec-mic_low_freq))
        self.fft_samples = (np.abs(np.fft.fft(self.get_samples()))[0:int(np.floor(self.sample_size/2))])/self.sample_size
        self.fft_samples[1:] = 2*self.fft_samples[1:]

        self.max_loc = np.argmax(self.fft_samples[low_freq_loc:]) + low_freq_loc
        return self.fft_samples
    
    # Divides Fourier transform into 8 octaves and returns a list of the max value in each octave
    def get_octaves(self, ft_data):
        octaves = []
        oct_end = len(ft_data)
        for i in range(1, 9, 1):
            octaves.append(np.amax(ft_data[int(oct_end/2):oct_end]))
            oct_end = int(oct_end / 2)
        octaves.reverse()
        octaves = np.array(octaves)
        return octaves

    # def plot_ft(self):
    #     plt.style.use('ggplot')
    #     plt.rcParams['font.size']=18
    #     fig = plt.figure(figsize=(13,8))
    #     ax = fig.add_subplot(111)
    #     plt.plot(self.f_vec,self.fft_samples)
    #     ax.set_ylim([0,2*np.max(self.fft_samples)])
    #     plt.xlabel('Frequency [Hz]')
    #     plt.ylabel('Amplitude [Pa]')
    #     ax.set_xscale('log')
    #     plt.grid(True)
    #
    #     # # max frequency resolution
    #     # plt.annotate(r'$\Delta f_{max}$: %2.1f Hz' % (self.sample_rate/(2*self.sample_size)),xy=(0.7,0.92),\
    #     #             xycoords='figure fraction')
    #
    #     # annotate peak frequency
    #     annot = ax.annotate('Freq: %2.1f'%(self.f_vec[self.max_loc]),xy=(self.f_vec[self.max_loc],self.fft_samples[self.max_loc]),\
    #                         xycoords='data',xytext=(0,30),textcoords='offset points',\
    #                         arrowprops=dict(arrowstyle="->"),ha='center',va='bottom')
    #
    #     plt.show()


class SpectroAnim(TreeAnimator):
    def __init__(self, coords_path):
        super(SpectroAnim, self).__init__(coords_path)
        self.coords_path = coords_path

    # Get the total height of the lights distribution
    # Divide into octaves
    def octave_heights(self):
        NUM_BANDS = 8

        coords = self._xyz_coords.copy()

        # Get heights and divide into octaves
        # z is the height               
        self.min_height = coords[:,2].min()                      
        self.max_height = coords[:,2].max()
        amp_height = self.max_height - self.min_height
        self.band_height = amp_height / NUM_BANDS

        self.octave_height = [self.min_height + i * self.band_height for i in range(1, 9, 1)]

    def initialize_animation(self):
        # This function only gets called once at the start of the animation.
        # Do anything in here that is related to the setup of your animation, initializing variables, connecting to APIs, whatever
        self.ft = FT()

        # Get min and max heights of lights
        self.octave_heights()

    def calculate_colors(self, xyz_coords, start_time):
        # this function gets called every few milliseconds, and it's purpose is to return the colors we want each light to be.

        # Get samples from the mic
        self.ft.get_samples()
        # Perform fft and format as octaves
        octave_powers = self.ft.get_octaves(self.ft.get_ft())
        # Normalize octaves between 0-1
        octave_powers = octave_powers / np.linalg.norm(octave_powers)
        # Turn normalization into a colour brightness 
        octave_colors = octave_powers * 255

        # Init array for color data
        colors = np.zeros((self.NUM_LIGHTS, 3), dtype=np.uint8)

        colors = np.ndarray((self.NUM_LIGHTS, 3), dtype=np.uint8)
        for i, light in enumerate(xyz_coords):
            for j, height in enumerate(self.octave_height):
                if light[2] < height:
                    colors[i] = np.array([octave_colors[j], 0, 0])
                    break


        # the arguments passed into this function are xyz_coords. this is an Nx3 array, where N is the number of lights (500)
        # start_time is the datetime straight after the initialize_animation() is ran
        # we can calculate how many seconds the animation has been running for by:
        num_seconds_since_start = (datetime.now() - start_time).total_seconds()
        # each time this function is called, num_seconds_since_start will get larger, and this can be used to control and time animations

        # This function is expected to return colors. colors are in RGB and in the range 0-255, for example black is [0,0,0], white is [255,255,255], red is [255,0,0]
        return colors

if __name__ == "__main__":
    coords_path = "./coordinates/sample_coords.csv"
    anim = SpectroAnim(coords_path)

    anim.animation_loop()
    # ft = FT()
