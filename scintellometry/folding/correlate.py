"""
 prawn modules for mpi-h5py
  gcc/4.8.2 hdf5/1.8.12-gcc-4.8.2-openmpi-1.6.5
  fftw/3.3.3-gcc-4.8.2-openmpi-1.6.5
  openmpi/1.6.5-gcc-4.8.2
  python/2.7.6
  git/1.8.2.1

"""
from __future__ import division, print_function

from fractions import Fraction
import numpy as np
import os
import astropy.units as u
from astropy.time import Time
import h5py

try:
    # do *NOT* use on-disk cache; blue gene doesn't work; slower anyway
    # import pyfftw
    # pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.scipy_fftpack import (rfft, rfftfreq, irfft,
                                                 fft, ifft, fftfreq)
    _fftargs = {'threads': os.environ.get('OMP_NUM_THREADS', 2),
                'planner_effort': 'FFTW_ESTIMATE'}
except(ImportError):
    print("Consider installing pyfftw: https://github.com/hgomersall/pyFFTW")
    # use FFT from scipy, since unlike numpy it does not cast up to complex128
    from scipy.fftpack import rfft, rfftfreq, irfft, fft, ifft, fftfreq
    _fftargs = {}

dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc
_fref = 150. * u.MHz  # ref. freq. for dispersion measure


def correlate(fh1, fh2, dm, nchan, ngate, ntbin, nt, ntw,
              dedisperse='incoherent', rfi_filter_raw=None, fref=_fref,
              save_xcorr=True, do_foldspec=True, phasepol=None,
              do_waterfall=True,
              t0=None, t1=None, comm=None, verbose=2):
    """
    fh1 : file handle of first data stream
    fh2 : file handle of second data stream
    dm :
    nchan :
    t0 : start time (isot) x-corr
         [None] start at common beginning of (fh1, fh2)
    t1 : end time of (isot) x-corr
         [None] end at common ending of (fh1, fh2)
    comm : MPI communicator or None

    """
    fhs = [fh1, fh2]
    if comm is None:
        rank = 0
        size = 1
    else:
        rank = comm.rank
        size = comm.size

    # find nearest starttime landing on same sample
    if t0 is None:
        t0 = max(fh1.time0, fh2.time0)
        print("Starting at %s" % t0)
    t0 = Time(t0, scale='utc')
    t1 = Time(t1, scale='utc')

    # find time offset between the two fh's, accomodating the relative phase
    # delay of the pulsar (the propagation delay)
    phases = [phasepol[i]((t0 - fhs[i].time0).sec) for i in [0, 1]]
    F0 = np.mean([phasepol[i].deriv(1)((t0 - fhs[i].time0).sec)
                  for i in [0, 1]])
    # propagation delay offset from fh1
    dts = [0. * u.s, np.diff(phases)[0] / F0 * u.s]
    if rank == 0:
        print("Will read fh2 ({0}) {1} ahead of fh1 ({2}) "
              "for propagation delay".format(fh2.telescope,
                                             dts[1].to(u.millisecond),
                                             fh1.telescope))

    # prep the fhs for xcorr stream, setting up channelization, dedispersion...
    for i, fh in enumerate(fhs):
        fh.seek(t0)
        # byte offset for propagation delay
        fh.prop_delay = int(round(dts[i] / fh.dtsample)) * fh.recordsize
        fh.dt1 = (1. / fh.samplerate).to(u.s)
        fh.this_nskip = fh.nskip(t0)
        if rank == 1:
            return None
        # set up FFT functions: real vs complex fft's
        if fh.nchan > 1:
            fh.thisfft = fft
            fh.thisifft = ifft
            fh.thisfftfreq = fftfreq
        else:
            fh.thisfft = rfft
            fh.thisifft = irfft
            fh.thisfftfreq = rfftfreq

        # pre-calculate time delay due to dispersion in coarse channels
        # LOFAR data is already channelized
        if fh.nchan > 1:
            fh.freq = fh.frequencies
        else:
            if fh.fedge_at_top:
                fh.freq = fh.fedge\
                    - fh.thisfftfreq(nchan * 2, fh.dt1.value) * u.Hz
            else:
                fh.freq = fh.fedge\
                    + fh.thisfftfreq(nchan * 2, fh.dt1.value) * u.Hz
            # [::2] sets frequency channels to numerical recipes ordering
            # or, rfft has an unusual ordering
            fh.freq = fh.freq[::2]

        # sort channels from low --> high frequency
        if np.diff(fh.freq.value).mean() < 0.:
            if rank == 0 and verbose > 1:
                print("Will frequency-sort {0} data before x-corr"
                      .format(fh.telescope))
            fh.freqsort = True
        else:
            fh.freqsort = False

        fh.dt = (dispersion_delay_constant * dm *
                 ( 1./fh.freq**2 - 1./fref**2) ).to(u.s).value

        # number of time bins to np.roll the channels for incoherent dedisperse
        if dedisperse == 'incoherent':
            fh.ndt = (fh.dt / fh.dtsample.to(u.s).value)
            fh.ndt = -1 * np.rint(fh.ndt).astype(np.int)

        elif dedisperse in ['coherent', 'by-channel']:
            # pre-calculate required turns due to dispersion
            if fh.nchan > 1:
                fcoh = (fh.freq[np.newaxis, :] + fftfreq(fh.ntint(nchan),
                        fh.dtsample.value)[:, np.newaxis] * u.Hz)
            else:
                if fh.fedge_at_top:
                    fcoh = fh.fedge - fh.thisfftfreq(nchan*2*fh.ntint(nchan),
                                                     fh.dt1.value) * u.Hz
                else:
                    fcoh = fh.fedge + fh.thisfftfreq(nchan*2*fh.ntint(nchan),
                                                     fh.dt1.value) * u.Hz

            #set frequency relative to which dispersion is coherently corrected
            if dedisperse == 'coherent':
                _fref = fref
            else:
                #fref = np.round((fcoh * fh.dtsample).to(1).value)/fh.dtsample
                _fref = np.repeat(fh.freq.value, fh.ntint(nchan))*fh.freq.unit
            # (check via eq. 5.21 and following in
            # Lorimer & Kramer, Handbook of Pulsar Astrono
            dang = (dispersion_delay_constant * dm * fcoh *
                    (1./_fref-1./fcoh)**2) * 360. * u.deg

            if fh.thisfftfreq is rfftfreq:
                # order of frequencies is r[0], r[1],i[1],...r[n-1],i[n-1],r[n]
                # for 0 and n need only real part, but for 1...n-1 need real, imag
                # so just get shifts for r[1], r[2], ..., r[n-1]
                dang = dang.to(u.rad).value[1:-1:2]
            else:
                dang = dang.to(u.rad).value

            fh.dd_coh = np.exp(dang * 1j).conj().astype(np.complex64)
    #### done fh setup ###

    ## xcorr setup
    # data-reading params (to help read in same-size time chunks and average
    # onto the same time-and-frequency grids)
    (Rf, Tf, NUf, fkeep, freqs, rows) = data_averaging_coeffs(fh1, fh2)
    nrows = int(min(rows[0] * Tf[1] / Tf[0], rows[1] * Tf[0] / Tf[1]))

    # summarize the (re)sampling
    if rank == 0:
        tmp = fh1.dtsample.to(u.s).value * fh1.blocksize / fh1.recordsize*Rf[0]
        print("\nReading {0} blocks of fh1, {1} blocks of fh2, "
              "for equal timeblocks of {2} sec ".format(Rf[0], Rf[1], tmp))

        if rank == 0 and verbose > 1:
            tmp = np.diff(freqs).mean()
            print("Averaging over {0} channels in fh1, {1} in fh2, for equal "
                  "frequency bins of {2} MHz".format(NUf[0], NUf[1], tmp))
            tmp = fh1.dtsample.to(u.s).value*Tf[0]
            print("Averaging over {0} timesteps in fh1, {1} in fh2, for equal "
                  "samples of {2} s\n".format(Tf[0], Tf[1], tmp))

        # check if we are averaging both fh's
        if rank == 0 and np.all(np.array(Tf) != 1):
            txt = "Note, averaging both fh's in time to have similar sample "\
                  "size. You may want to implement interpolation, or think "\
                  "more about this situation"
            print(txt)
        if rank == 0 and np.all(np.array(NUf) != 1):
            txt = "Note, averaging both fh's in freq to have similar sample "\
                  "size. You may want to implement interpolation, or think "\
                  "more about this situation"
            print(txt)

    # initialize the folded spectrum and waterfall
    nchans = min([len(fh.freq[fkeep[i]] / NUf[i]) for i, fh in enumerate(fhs)])
    foldspec = np.zeros((nchans, ngate, ntbin))
    icount = np.zeros((nchans, ngate, ntbin), dtype=np.int64)
    nwsize = min(nt * fh1.ntint(nchan) // ntw, nt * fh2.ntint(nchan) // ntw)
    waterfall = np.zeros((nchans, nwsize))

    if save_xcorr:
        # output dataset
        outname = "{0}{1}_{2}_{3}.hdf5".format(
            fh1.telescope[0], fh2.telescope[0], t0, t1)
        # mpi doesn't like colons
        outname = outname.replace(':', '')
        fcorr = h5py.File(outname, 'w')  # , driver='mpio', comm=comm)
        ## create the x-corr output file
        # save the frequency grids to help with future TODO: interpolate onto
        # same frequency grid. For now the frequencies fall within same bin
        if rank == 0 and verbose:
            print("Saving x-corr to %s\n" % outname)
        fcorr.create_dataset('freqs', data=np.hstack([f.to(u.MHz).value
                                                      for f in freqs]))

        # the x-corr data [tsteps, channels]
        dset = fcorr.create_dataset('corr', (nrows, freqs[0].size),
                                    dtype='complex64', maxshape=(None, nchan))
        dset.attrs.create('dedisperse', data=str(dedisperse))
        dset.attrs.create('tsample',
                          data=[fhs[i].dtsample.to(u.s).value * Tf[i]
                                for i in [0, 1]])
        dset.attrs.create('chanbw', data=np.diff(freqs).mean())

    # start reading the data
    # this_nskip moves to 't0', rank is for MPI
    idx = rank
    raws = [fh.seek_record_read((fh.this_nskip + idx * Rf[i])
                                * fh.blocksize - fh.prop_delay,
                                fh.blocksize * Rf[i])
            for i, fh in enumerate(fhs)]
    endread = False
    print("read step (idx), fh1.time(), fh2.time() ")
    print("\t inclues {0} propagation delay".format(dts[1]))
    while np.all([raw.size > 0 for raw in raws]):
        if verbose:
            print("idx",idx, fh1.time(), fh2.time())

        vals = raws
        chans = [None, None]
        tsamples = [None, None]
        isrs = [None, None]

        # prep the data (channelize, dedisperse, ...)

        for i, fh in enumerate(fhs):
            if rfi_filter_raw is not None:
                raws[i], ok = rfi_filter_raw(raws[i], nchan)

            if fh.telescope == 'aro':
                vals[i] = raws[i].astype(np.float32)
            else:
                vals[i] = raws[i]

            if dedisperse in ['coherent', 'by-channel']:
                fine = fh.thisfft(vals[i], axis=0, overwrite_x=True,
                                  **_fftargs)
                if fh.thisfft is rfft:
                    fine_cmplx = fine[1:-1].view(np.complex64)
                    # overwrites parts of fine, as intended
                    fine_cmplx *= fh.dd_coh
                else:
                    fine *= dd_coh
                vals[i] = fh.thisifft(fine, axis=0, overwrite_x=True,
                                      **_fftargs)

            if fh.nchan == 1:
                # ARO data should fall here
                chans[i] = fh.thisfft(vals[i].reshape(-1, nchan * 2), axis=-1,
                                      overwrite_x=True, **_fftargs)
            else:  # lofar and gmrt-phased are already channelised
                chans[i] = vals[i]

            # dedisperse on original (raw) time/freq grid
            # TODO: profile for speedup
            if dedisperse == 'incoherent':
                for ci, v in enumerate(fh.ndt):
                    chans[i][..., ci] = np.roll(chans[i][..., ci], v, axis=0)

            # average onto same time grid
            chans[i] = chans[i].reshape(Tf[i], chans[i].shape[0] / Tf[i], -1)\
                      .mean(axis=0)

            # average onto same freq grid
            chans[i] = chans[i][..., fkeep[i]]
            chans[i] = chans[i].reshape(-1, chans[i].shape[1] / NUf[i],
                                        NUf[i]).mean(axis=-1)

            # current sample positions in stream
            # (each averaged onto same time grid)
            isrs[i] = idx * rows[i] + np.arange(rows[i])
            tsamples[i] = (fh.this_nskip * fh.dtsample * fh.ntint(nchan)
                           + isrs[i] * fh.dtsample)
            tsamples[i] = tsamples[i].reshape(-1, Tf[i]).mean(axis=-1)

            # finally sort the channels low --> high (if necessary)
            # before x-correlating
            if fh.freqsort:
                # TODO: need to think about ordering
                chans[i] = chans[i][..., ::-1]

        # x-correlate
        xpower = chans[0] * chans[1].conjugate()

        if do_waterfall:
            # loop over corresponding positions in waterfall
            isr = idx * nrows + np.arange(nrows)
            for iw in range(isr[0] // ntw, isr[-1] // ntw + 1):
                if iw < nwsize:  # add sum of corresponding samples
                    waterfall[0:xpower.shape[1], iw] += \
                        np.abs(np.sum(xpower[isr // ntw == iw], axis=0))

        if do_foldspec:
            # time since start (average of the two streams)
            # TODO: think about this: take one stream, both, average, ...
            #tsample = np.mean(tsamples, axis=0)
            tsample = np.array(tsamples)

            # timeseries already dedispersed
            phase = phasepol[0](tsample[0])
            iphase = np.remainder(phase * ngate,
                                  ngate).astype(np.int)

            # bin in the time series: 0..ntbin-1
            ibin = idx * ntbin // nt
            for k in range(nchans):  # equally xpower.shape[1]
                foldspec[k, :, ibin] += np.bincount(iphase,
                                                    np.abs(xpower[:, k]),
                                                    ngate)
                icount[k, :, ibin] += np.bincount(iphase,
                                                  np.abs(xpower[:, k]) != 0.,
                                                  ngate)

        if save_xcorr:
            curshape = dset.shape
            nx = max(nrows * (idx + 1), curshape[0])
            dset.resize((nx + nrows, curshape[1]))
            # TODO: h5py mpio stalls here... turn off save_xcorr for mpirun
            dset[nrows * idx: nrows * (idx + 1)] = xpower

        # read in next dataset if we haven't hit t1 yet
        for fh in [fh1, fh2]:
            if (fh.time() - t1).sec > 0.:
                endread = True

        if endread:
            break
        else:
            idx += size
            raws = [fh.seek_record_read((fh.this_nskip + idx * Rf[i])
                                        * fh.blocksize - fh.prop_delay,
                                        fh.blocksize * Rf[i])
                    for i, fh in enumerate(fhs)]
    if save_xcorr:
        fcorr.close()
    return foldspec, icount, waterfall

def data_averaging_coeffs(fh1, fh2):
    """
    return the time-and-frequency averaging parameters
    which help get the data on the same grid(s)

    """
    ## read the two filestreams in chunks of equal time
    sr = Fraction(fh1.dtsample.to(u.s).value * fh1.blocksize
                  / fh1.recordsize)
    sr /= Fraction(fh2.dtsample.to(u.s).value * fh2.blocksize
                   / fh2.recordsize)
    sr = sr.limit_denominator(1000)
    nf1 = sr.denominator
    nf2 = sr.numerator

    ## used for re-sizing hdf5 x-corr output
    raw1_nrows = int(fh1.blocksize * nf1 / fh1.recordsize)
    raw2_nrows = int(fh2.blocksize * nf2 / fh2.recordsize)

    ## time averaging params
    Tavg = Fraction(raw1_nrows, raw2_nrows).limit_denominator(1000)
    Tden = Tavg.denominator
    Tnum = Tavg.numerator

    ## channel averaging params
    f1info = (fh1.freq.max(), fh1.freq.min(), len(fh1.freq),
              np.sign(np.diff(fh1.freq).mean()))
    f2info = (fh2.freq.max(), fh2.freq.min(), len(fh2.freq),
              np.sign(np.diff(fh2.freq).mean()))
    f1keep = (fh1.freq > max(f1info[1], f2info[1])) \
        & (fh1.freq < min(f1info[0], f2info[0]))
    f2keep = (fh2.freq > max(f1info[1], f2info[1])) \
        & (fh2.freq < min(f1info[0], f2info[0]))

    Favg = abs(Fraction(np.diff(fh1.freq.value).mean()
                        / np.diff(fh2.freq.value).mean()))
    Favg = Favg.limit_denominator(200)
    Fden = Favg.denominator
    Fnum = Favg.numerator
    # the frequencies we keep
    freq1 = fh1.freq[f1keep]
    freq1 = freq1.reshape(freq1.size / Fden, Fden).mean(axis=-1)
    freq2 = fh2.freq[f2keep]
    freq2 = freq2.reshape(freq2.size / Fnum, Fnum).mean(axis=-1)
    # sort low freq to high freq
    if f1info[3] < 0:
        freq1 = freq1[::-1]
    if f2info[3] < 0:
        freq2 = freq2[::-1]

    return ((nf1, nf2), (Tnum, Tden), (Fden, Fnum), (f1keep, f2keep),
            (freq1, freq2), (raw1_nrows, raw2_nrows))
