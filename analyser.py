import random
import sys
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sclib import SoundcloudAPI


class AudioAnalyzer:

    def __init__(self):
        self.frequencies_index_ratio = 0
        self.map_file = None
        self.time_index_ratio = 0
        self.spectrogram = None
        self.frequencies = None
        self.bass_times = []

    def load(self, file_name):

        self.map_file = file_name.split('.')[0] + '.txt'

        time_series, sample_rate = librosa.load(file_name)
        
        stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

        self.spectrogram = librosa.amplitude_to_db(stft, ref=np.max)

        self.audio_length = int(librosa.get_duration(y=time_series, sr=sample_rate))

        frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

        times = librosa.core.frames_to_time(np.arange(
            self.spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4)

        self.time_index_ratio = len(times) / times[len(times) - 1]

        self.frequencies_index_ratio = len(
            frequencies) / frequencies[len(frequencies) - 1]

    def get_decibel(self, target_time, freq):
        return self.spectrogram[int(freq * self.frequencies_index_ratio)][int(target_time * self.time_index_ratio)]

    def show(self):
        librosa.display.specshow(self.spectrogram, y_axis='log', x_axis='time')

        plt.title('Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()

    def print_spectrogram(self):
        for i in range(len(self.spectrogram)):
            print(len(self.spectrogram[i]), end='')
            for j in range(len(self.spectrogram[i])):
                print(self.spectrogram[i][j], end=' ')
            print()

    def analyze_spectrogram(self):

        bass = {"start": 50, "stop":150}
        # heavy_area = {"start": 120, "stop": 250}
        # heavy_area = {"start": 120, "stop": 250}
        # low_mids = {"start": 251, "stop": 2000}
        # high_mids = {"start": 2001, "stop": 6000}

        bass_sum = 0
        bass_flag = False
        freqs = [bass, """heavy_area, low_mids, high_mids"""]

        for time_index in np.arange(0.0, self.audio_length, 0.001):
            current_bass = 0
            for i in range(freqs[0]['start'], freqs[0]['stop']):
                current_bass += self.get_decibel(time_index, i)
            bass_sum += current_bass / (freqs[0]['stop'] - freqs[0]['start'])

        bass_threshold = bass_sum / (self.audio_length * 1000)

        bass_threshold = int((bass_threshold / 10) * 7)

        print(bass_threshold)

        for time_index in range(self.audio_length * 1000):
            milli_second = time_index / 1000.0
            bass_sum = 0
            for i in range(freqs[0]['start'], freqs[0]['stop']):
                bass_sum += self.get_decibel(milli_second, i)
            bass_avg = bass_sum / (freqs[0]['stop'] - freqs[0]['start'])
            if bass_avg > bass_threshold and not bass_flag:
                self.bass_times.append({'start': time_index})
                bass_flag = True
            elif bass_avg < bass_threshold and bass_flag:
                self.bass_times[len(self.bass_times) - 1]['end'] = time_index
                bass_flag = False

        avg_len = 0
        for bass in self.bass_times:
            bass['length'] = bass['end'] - bass['start']
            avg_len += bass['length']

        avg_len = int(((avg_len / len(self.bass_times)) / 10) * 8)
        self.bass_times = [x for x in self.bass_times if x['length'] >= avg_len]

        for bass in self.bass_times:
            print(f"start: {bass['start']},end: {bass['end']},length:{bass['length']}")

        self.generate_map()

    def generate_map(self):
        # Row length is based on music,
        # Tunnel length is 1 (player drop area)
        # Tunnel starting position is 5
        # When beat kicks in tunnel changes position,
        # By following tunnel player should feel the rhythm
        # Player Speed is 20 Map length = 20 * Music.Length
        map_length = 20 * self.audio_length
        position = 0
        last_position = 0
        positions = []
        for bass in self.bass_times:
            forward_to = int((bass['length'] * 20) / 1000)
            position = last_position + forward_to if random.randint(0, 1) > 0 else last_position - forward_to
            positions.append(
                {'occurs_at': int((bass['start'] * 20) / 1000), 'start': last_position, 'end': position, 'length': forward_to})
            last_position = position

        f = open(self.map_file,'w')
        for pos in positions:
            f.write(f"occurs_at:{pos['occurs_at']}, start:{pos['start']}, end:{pos['end']}, length:{pos['length']}\n")
        f.close()



def download_song(url):
    api = SoundcloudAPI()

    track = api.resolve(url)

    file_name = f'./{track.artist}_{track.title}.mp3'

    with open(file_name, 'wb+') as fp:
        track.write_mp3_to(fp)
    return file_name


def main(file_name):
    analyzer = AudioAnalyzer()
    analyzer.load(file_name)
    #analyzer.show()
    #analyzer.print_spectrogram()
    analyzer.analyze_spectrogram()


if __name__ == '__main__':
    if str(sys.argv[1]) == 'u':
        file_name = download_song(str(sys.argv[2]))
        main(file_name)
    else:
        main(sys.argv[2])
