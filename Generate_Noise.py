import sys
from random import sample
import time

import numpy as np
from joblib import Parallel, delayed
import shutil

sys.path.append("..")
    
from Configuration.Room_array_CFG import *
from Generate_Rirs_Cpu import gen_RIRs
import scipy.signal as ss
from Dsp.Basetool import *


array_pos = gen_array(array_type)
t60_list = np.load("../Configuration/T60.npy")
noise_length = 30
Menteenumb = 100


def generate_White_Noise(noise_len=noise_length, P=array_pos.T, fs=16000):
    # the size of both noise and y should be samples*channel
    channel = P.shape[1]
    noise_len = int(noise_len * fs)
    white_noise = np.zeros((channel, noise_len))
    for i in range(channel):
        white_noise[i] = np.random.randn(noise_len)
        white_noise[i] /= np.sqrt(np.sum(white_noise[i] ** 2.0))
    white_noise = white_noise * 0.1
    return white_noise


def generate_3D_Arbitrary_Diffuse_Noise(P_3M=array_pos.T, noise_Len=noise_length, n_Points=2048,
                                        fs=16000, c=343, n_type="spherical"):
    # choose the first microphone as origin
    M = P_3M.shape[1]
    P_Relative_3M = P_3M.copy()
    P_Relative_3M -= P_3M[:, 0, None]
    # in the frequency domain, nfft always be even
    nfft = int(2 ** np.ceil(np.log2(noise_Len * fs)))
    omega_F = 2 * np.pi * np.arange(int(nfft / 2) + 1) / nfft * fs
    V_N3 = np.zeros((n_Points, 3), dtype=np.complex_)
    # simulate the spatial distribution of spherical and cylinder
    if n_type == "spherical":
        # generate N points that are near-uniformly distributed over s^2
        h = -1 + 2.0 * np.arange(n_Points) / (n_Points - 1)
        phi = np.arccos(h)
        theta = np.zeros(n_Points)
        for k in range(1, n_Points - 1):
            theta[k] = (theta[k - 1] + 3.6 / np.sqrt(n_Points * (1 - h[k] ** 2.0))) % (2.0 * np.pi)
        # transform matrix (the inner product with array geometry will be the time delay)
        V_N3[:, 0] = np.cos(theta) * np.sin(phi)
        V_N3[:, 1] = np.sin(theta) * np.sin(phi)
        V_N3[:, 2] = np.cos(phi)
    elif n_type == "cylinder":
        phi = 2.0 * np.pi * np.arange(n_Points) / n_Points
        V_N3[:, 0] = np.cos(phi)
        V_N3[:, 1] = np.sin(phi)

    # calculate the time delay from source N to microphone M
    Delta_NM = (V_N3 @ P_Relative_3M) / c

    # generate N point sources at each frequency bin
    X_prime_NF = np.random.randn(n_Points, int(nfft / 2 + 1)) + 1j * np.random.randn(n_Points, int(nfft / 2 + 1))
    # received signal
    X_MF = np.zeros((M, int(nfft / 2 + 1)), dtype=X_prime_NF.dtype)
    # received signal at the first channel
    X_MF[0] = np.sum(X_prime_NF, axis=0)
    # cut into frequency blocks (since not enough memory can be provided by my laptop)
    block_Size = 1
    n_Blocks = int((nfft / 2 + 1) // block_Size)
    n_BlockFrames = int(n_Blocks * block_Size)
    for block_Index in range(n_Blocks):
        # calculate the received sensor signal, Nx1xF * (N*M*1 * 1*1*F), element-wise product
        X_MF[1:, block_Index * block_Size:(block_Index + 1) * block_Size] = np.sum(
            X_prime_NF[:, None, block_Index * block_Size:(block_Index + 1) * block_Size] *
            np.exp(-1j * Delta_NM[:, :, None] * omega_F[None, None,
                                                block_Index * block_Size:(block_Index + 1) * block_Size:]), axis=0,
            keepdims=False)[1:]
    # the last block
    X_MF[1:, n_BlockFrames:] = np.sum(X_prime_NF[:, None, n_BlockFrames:] *
                                      np.exp(-1j * Delta_NM[:, :, None] * omega_F[None, None, n_BlockFrames:]), axis=0,
                                      keepdims=False)[1:]
    X_MF /= np.sqrt(n_Points)
    # transform back to the time domain
    X_d_MF = np.zeros((M, nfft), dtype=X_MF.dtype)
    X_d_MF[:, 0] = np.sqrt(nfft) * X_MF[:, 0].real
    X_d_MF[:, 1:int(nfft / 2)] = np.sqrt(int(nfft / 2)) * X_MF[:, 1:-1]
    X_d_MF[:, int(nfft / 2)] = np.sqrt(nfft) * X_MF[:, int(nfft / 2)].real
    X_d_MF[:, int(nfft / 2) + 1:] = np.sqrt(int(nfft / 2)) * np.conj(X_MF[:, ::-1][:, 1:-1])
    X_d_MT = ((np.fft.ifft(X_d_MF, n=int(nfft))).real * 1e-2)[:, :noise_Len * fs]
    return X_d_MT


def generate_Once_Diffuse_Noise(cnt=None, diffuse_dir=None):
    diffuse_noise = generate_3D_Arbitrary_Diffuse_Noise()
    diffuse_file = diffuse_dir + str(cnt + 1) + ".wav"
    sf.write(diffuse_file, diffuse_noise.T, samplerate=16000)


def generate_Once_White_Noise(cnt=None, white_dir=None):
    white_noise = generate_White_Noise()
    white_file = white_dir + str(cnt + 1) + ".wav"
    sf.write(white_file, white_noise.T, samplerate=16000)


def generate_Once_BG_Noise(cnt=None, bg_dir=None, INF_RIR_path=None,
                           INF_list=None, Inf_len=noise_length):
    if not os.path.exists(bg_dir):
        os.makedirs(bg_dir)
    Inf_RIRs = np.load(INF_RIR_path)
    N_Inf = Inf_RIRs.shape[0]
    Inf_spk_list = sample(INF_list, N_Inf)
    INF_img_list = list()
    for n_index in range(N_Inf):
        INF, _ = librosa.load(Inf_spk_list[n_index], sr=16000)
        INF, _, _ = tailor_dB_VAD_FS(INF, -20)
        INF = INF - np.mean(INF)
        RIR = Inf_RIRs[n_index]
        INF_Img = np.squeeze(ss.fftconvolve(RIR[:, None, :], INF[:, None, None]))
        INF_img_list.append(INF_Img)
    background = np.sum(np.asarray(INF_img_list), axis=0)
    background = background[-Inf_len*fs-1:-1, :]
    print(background.shape)
    sf.write(os.path.join(bg_dir, str(cnt + 1) + ".wav"), background, samplerate=16000)


if __name__ == "__main__":
    num_workers = 20
    noise_dir = "../Dataset/Noise/"
    if not os.path.exists(noise_dir):
        os.makedirs(noise_dir)

    diffuse_dir = "../Dataset/Noise/Diffuse_noise/"
    if not os.path.exists(diffuse_dir):
        os.makedirs(diffuse_dir)

    white_dir = "../Dataset/Noise/White_noise/"
    if not os.path.exists(white_dir):
        os.makedirs(white_dir)

    INFs_RIR_path = "../Dataset/SS&SE/RIRs/INFs/"
    if os.path.exists(INFs_RIR_path):
        shutil.rmtree(INFs_RIR_path)
    os.makedirs(INFs_RIR_path)
    Parallel(n_jobs=num_workers)(
        delayed(gen_RIRs)(
            T60_int=t60_list[cnt], RIR_save_path=INFs_RIR_path, cnt=cnt, SOI_or_INF="INF") for cnt in range(Menteenumb))

    bg_dir = "../Dataset/SS&SE/Background/"
    if os.path.exists(bg_dir):
        shutil.rmtree(bg_dir)
    os.mkdir(bg_dir)
    INFs_filename_path = "../Dataset/XJNoise/"
    INF_list = [os.path.join(INFs_filename_path, i) for i in listfile(INFs_filename_path, postfix=".wav")]

    time.sleep(0.3)
    INFs_RIR_filename_path = "../Dataset/SS&SE/RIRs/INFs/"
    INFs_RIR_filename_list = listfile(INFs_RIR_filename_path, postfix=".npy")
    Parallel(n_jobs=num_workers)(
        delayed(generate_Once_BG_Noise)(cnt, bg_dir, INF_RIR_path=os.path.join(INFs_RIR_filename_path,
                                                                               INFs_RIR_filename_list[cnt]),
                                        INF_list=INF_list) for cnt in range(100))

    time.sleep(0.3)
    Parallel(n_jobs=num_workers)(
        delayed(generate_Once_White_Noise)(
            cnt, white_dir) for cnt in range(Menteenumb)
    )

    time.sleep(0.3)
    Parallel(n_jobs=num_workers)(
        delayed(generate_Once_Diffuse_Noise)(
            cnt, diffuse_dir) for cnt in range(Menteenumb)
    )
