import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensound

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
def load_img(path, d=None):
    fractal = imread(path)
    if d and d%2==0:
        fractal = fractal[::d, ::d]
    image_map = np.zeros((len(fractal)**2, 4), dtype=int)
    image_map[:, 0] = hilbert_approx(len(fractal)).flatten()
    image_map[:, 1] = fractal[:, :, 0].ravel()
    image_map[:, 2] = fractal[:, :, 1].ravel()
    image_map[:, 3] = fractal[:, :, 2].ravel()
    image_map[np.argsort(image_map[:, 0])]
    return image_map, fractal

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

import argparse

def main():
    parser = argparse.ArgumentParser(description='Transform input image to simple music.')
    parser.add_argument(
    '--input', '-i', type=str, default='foret.jpeg',
        help="Input file (default: 'foret.jpeg').")
    parser.add_argument(
        '-n', type=int, default=1,
        help="matrix dim")
    args = parser.parse_args()
    image_cible = args.input

    img, img_raw = load_img(image_cible, args.n)
    print(len(img_raw))
    plt.ion()
    plt.figure()
    #imgplot = plt.imshow(img_raw[::args.n, ::args.n])
    imgplot = plt.imshow(img_raw)
    #plt.show()
    plt.pause(0.001)

    notes = load_notes()


    #freq_seq = [get_freq(notes, img[i, 1], taux_min=40, taux_max=60) for i in range(len(img))]
    notes_seq = [get_note(notes, img[i, 1], taux_min=40, taux_max=60) for i in range(len(img))]
    # 2) set its parameters using sequences to be iterated through
    wave = gensound.Sine(' '.join(notes_seq[:100]), duration=0.25e3)
    print(notes_seq)
    wave.play()

if __name__ == '__main__':
    main()
