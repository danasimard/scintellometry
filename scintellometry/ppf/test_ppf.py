from __future__ import division

import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pylab as plt
import astropy.units as u

import ppf

if __name__ == '__main__':
    fppf = ppf.FastPPF()
    ippf = ppf.FastiPPF()
    size, shape = fppf.weights.size, fppf.weights.shape
    samplerate = 200*u.MHz
    t = (np.arange(-size + 0., size*2.) / samplerate).to(u.s)
    # put in frequencies near and away from band centres
    omega = np.array([700.,
                      800.49,
                      850. - 3.5/16.,
                      900.5,
                      600. + 7./16.,
                      650. - 6./16.]) / 1024. * u.cycle * samplerate
    timedata = np.sin(omega[:, np.newaxis] * t + 0.1*u.cycle).sum(0)
    # possibly add delta function
    # timedata[size+100] = 1e10
    # for comparison, straight FT
    unfiltered = rfft(timedata[size//2:-size//2].reshape(-1, shape[-1]) *
                      fppf.weights.sum(0))

    # polyphase filter is filled in round-robin fashion, 1024 at a time,
    # 16 times.  So, just reshape(shape) gets that done.
    ppfdata = np.array([fppf(timedata[s-size//2:s+size//2].reshape(shape))
                        for s in range(size//2, len(timedata)-size//2,
                                        shape[-1])])
    # check convolution -> same as multiplication in Fourier domain
    timedata_fft = rfft(timedata.reshape(-1, shape[-1]), axis=0)
    ppf_weights = np.zeros((timedata.size // shape[-1], shape[-1]))
    ppf_weights[:shape[0]] = fppf.weights
    ppf_fft = rfft(ppf_weights, axis=0)
    ppfdata_via_fft = np.fft.irfft(timedata_fft*ppf_fft.conj(), axis=0)
    # only first bit is reliable (further bits used rolled-around data)
    ppfdata_via_fft = ppfdata_via_fft[:ppfdata.shape[0]]
    assert np.all(np.isclose(ppfdata_via_fft, ppfdata))

    filtered = rfft(ppfdata, axis=-1)
    freq = samplerate - np.fft.rfftfreq(shape[-1], d=1./samplerate)
    plt.plot(np.arange(1024., 511, -1), np.abs(unfiltered.T))
    plt.plot(np.arange(1024., 511, -1), np.abs(filtered.T))

    # now have lofar-like filterbank data.  See if we can recover original
    inverse = irfft(filtered)
    # check that inverse fourier transform does the right thing
    assert np.all(np.isclose(inverse, ppfdata))
    # ippf.weights /= (fppf.weights.sum(0, keepdims=True) *
    #                  ippf.weights.sum(0, keepdims=True))
    timeback = np.array([ippf(inverse[s-8:s+8].reshape(shape))
                        for s in range(8, inverse.shape[0]-8)])
    ippf_weights = np.zeros_like(inverse)
    ippf_weights[:shape[0]] = ippf.weights
    fppf_weights = np.zeros_like(inverse)
    fppf_weights[:shape[0]] = fppf.weights
    timeback_via_fft_i = irfft(rfft(ppfdata_via_fft, axis=0) *
                               rfft(ippf_weights, axis=0).conj(), axis=0)[0]
    timeback_via_fft_f = irfft(rfft(ppfdata_via_fft, axis=0) /
                               rfft(fppf_weights, axis=0).conj(), axis=0)[0]
    assert np.all(np.isclose(timeback_via_fft_i, timeback[0]))
    plt.clf()
    plt.plot(timeback[0])
    # plt.plot(inverse.flatten()[size//2-1024:size//2])
    plt.plot(timedata.reshape(-1, 1024)[15].value)
    # good to ~0.06/2 = 3%

    if False:
        plt.plot(np.angle(np.fft.rfft(fppf.weights, axis=0).conj().T))
        plt.plot(np.angle(np.fft.rfft(ippf.weights, axis=0).conj().T[::-1]))
        # makes sense, some deviations that do not
