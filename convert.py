import numpy as np
import matplotlib.pyplot as plt

def hilbert_approx(size=2, start=0, quarter='n'):
    if size == 2:
        mat = np.array([[1, 2],[0, 3]]) + start
    else:
        mat = np.zeros((size, size))
        half = int(size / 2)
        mat[half:size, 0:half] = hilbert_approx(half, start=start, quarter='so')
        mat[0:half, 0:half] = hilbert_approx(half, start=start + half**2)
        mat[0:half, half:size] = hilbert_approx(half, start=start + 2 * (half**2))
        mat[half:size, half:size] = hilbert_approx(half, start=start + 3 * (half**2), quarter='se')
    if quarter == 'so':
        return np.fliplr(np.rot90(mat, k=1)) + start
    elif quarter == 'se':
        return np.flipud(np.rot90(mat, k=1)) + start
    else:
        return mat + start

from matplotlib.image import imread
def load_img(path):
    fractal = imread(path)
    image_map = np.zeros((len(fractal)**2, 4), dtype=int)
    image_map[:, 0] = hilbert_approx(len(fractal)).flatten()
    image_map[:, 1] = fractal[:, :, 0].ravel()
    image_map[:, 2] = fractal[:, :, 1].ravel()
    image_map[:, 3] = fractal[:, :, 2].ravel()
    image_map[np.argsort(image_map[:, 0])]
    return image_map

import pandas as pd
def load_notes():
    return pd.read_csv('notes.csv', delimiter='\t', names=['note', 'freq'])

def get_note(df, pixel, taux_min=0, taux_max=100):
  taille = len(df)
  val_min = taux_min * taille / 100
  val_max = taux_max * taille / 100

  return df['note'][int(pixel * (val_max-val_min - 1) / 255 + val_min)]

def get_freq(df, pixel, taux_min=0, taux_max=100):
    taille = len(df)
    val_min = taux_min * taille / 100
    val_max = taux_max * taille / 100
    return df['freq'][int(pixel * (val_max-val_min - 1) / 255 + val_min)]

def main():
    img = load_img('foret.jpeg')
    notes = load_notes()
    import music as M
    T = M.tables.Basic()
    H = M.utils.H


    # 1) start a ѕynth
    b = M.core.Being()

    freq_seq = [get_freq(notes, img[i, 1], taux_min=40, taux_max=60) for i in range(len(img))]
    # 2) set its parameters using sequences to be iterated through
    print(freq_seq[7000:9000])
    #b.fv_ = [0, 1,5,15,150,1500,15000]  # vibrato frequency
    b.fv_ = [0]  # vibrato frequency
    b.nu_ = [0]  # vibrato depth in semitones (maximum deviation of pitch)
    b.f_ = freq_seq[7000:9000]
    b.d_ = [1 / 4] * len(b.f_)

    # 3) render the wavfile
    b.render(120, 'aMusicalSound.wav')  #
    # seq = []
    # for i in range(len(img)):
    #     seq.append(get_note(notes, img[i, 1]))
    # gammes = [note[-1] for note in seq]
    # print(gammes)
    # print(len(seq), seq[:300])
if __name__ == '__main__':
    main()
