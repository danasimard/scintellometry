""" FFT and Fold data """
from __future__ import division, print_function

from inspect import getargspec
import numpy as np
import os
import astropy.units as u

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.scipy_fftpack import rfft, rfftfreq, irfft, fft, ifft, fftfreq
    _fftargs = {'threads': os.environ.get('OMP_NUM_THREADS', 2),
                'planner_effort': 'FFTW_ESTIMATE'}
except(ImportError):
    print("Consider installing pyfftw: https://github.com/hgomersall/pyFFTW")
    # use FFT from scipy, since unlike numpy it does not cast up to complex128
    from scipy.fftpack import rfft, rfftfreq, irfft, fft, ifft, fftfreq
    _fftargs = {}

from fromfile import fromfile

dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc

def fold(fh, comm, dtype, samplerate, fedge, fedge_at_top, nchan,
         nt, ntint, nskip, ngate, ntbin, ntw, dm, fref, phasepol,
         dedisperse='incoherent',
         do_waterfall=True, do_foldspec=True, verbose=True,
         progress_interval=100, rfi_filter_raw=None, rfi_filter_power=None):
    """
    FFT data, fold by phase/time and make a waterfall series
    
    Parameters
    ----------
    fh : file handle
        handle to file holding voltage timeseries
    comm: MPI communicator or None
    dtype : numpy dtype or '4bit' or '1bit'
        way the data are stored in the file
    samplerate : float
        rate at which samples were originally taken and thus double the
        band width (frequency units)
    fedge : float
        edge of the frequency band (frequency units)
    fedge_at_top: bool
        whether edge is at top (True) or bottom (False)
    nchan : int
        number of frequency channels for FFT
    nt, ntint : int
        total number nt of sets, each containing ntint samples in each file
        hence, total # of samples is nt*ntint, with each sample containing
        a single polarisation
    nskip : int
        number of records (ntint * nchan * 2 / 2 bytes) to skip
    ngate, ntbin : int
        number of phase and time bins to use for folded spectrum
        ntbin should be an integer fraction of nt
    ntw : int
        number of time samples to combine for waterfall (does not have to be
        integer fraction of nt)
    dm : float
        dispersion measure of pulsar, used to correct for ism delay
        (column number density)
    fref: float
        reference frequency for dispersion measure
    phasepol : callable
        function that returns the pulsar phase for time in seconds relative to
        start of part of the file that is read (i.e., ignoring nhead)
    dedisperse : None or string
        None, 'incoherent', 'coherent', 'by-channel'
    do_waterfall, do_foldspec : bool
        whether to construct waterfall, folded spectrum (default: True)
    verbose : bool or int
        whether to give some progress information (default: True)
    progress_interval : int
        Ping every progress_interval sets
        
    """
    if comm is None:
        rank = 0
        size = 1
    else:
        rank = comm.rank
        size = comm.size

    # initialize folded spectrum and waterfall
    foldspec = np.zeros((nchan, ngate, ntbin))
    icount = np.zeros((nchan, ngate, ntbin), dtype=np.int64)
    nwsize = nt*ntint//ntw
    waterfall = np.zeros((nchan, nwsize))

    count = nchan*ntint
    itemsize = fh.itemsize
#    recsize = count*itemsize
    if verbose and rank == 0:
        print('Reading from {}'.format(fh))

    if nskip > 0:
        if verbose and rank == 0:
            print('Skipping {0} records = {1} bytes'
                  .format(nskip, nskip*fh.recsize))
        if size == 1:
            # MPI: only skip here if we are not threaded, otherwise seek in for-loop
            fh.seek(nskip * count * itemsize)

    # LOFAR data is already channelized
    if hasattr(fh, 'fwidth'):
       dtsample = (1./fh.fwidth).to(u.s)
       dt1 = dtsample
    else:
        #ARO data
        dt1 = (1./samplerate).to(u.s)
        # need 2*nchan real-valued samples for each FFT
        dtsample = nchan * 2 * dt1
    
    if fh.telescope == 'gmrt':
        # but include 2*nchan real-valued samples used for each FFT
        # (or, equivalently, real and imag for channels)
        tstart = dt1 * nskip * count * itemsize
    elif fh.telescope in ['aro', 'lofar']:
        # need 2*nchan real-valued samples for each FFT for ARO
        tstart = dtsample * ntint * nskip

    # set up FFT functions: real vs complex fft's
    if fh.telescope == 'aro':
        thisfft = rfft
        thisifft = irfft
        thisfftfreq = rfftfreq
    else:
        thisfft = fft
        thisifft = ifft
        thisfftfreq = fftfreq

    # pre-calculate time delay due to dispersion in coarse channels
    # LOFAR data is already channelized
    if hasattr(fh, 'freqs'):
        freq = fh.freqs
    else:
        if fedge_at_top:
            freq = fedge - thisfftfreq(nchan*2, dt1.value) * u.Hz
        else:
            freq = fedge + thisfftfreq(nchan*2, dt1.value) * u.Hz
  
    if fh.telescope == 'aro':
        # ARO data
        # [::2] sets frequency channels to numerical recipes ordering
        # or, rfft has an unusual ordering
        dt = (dispersion_delay_constant * dm *
                 (1./freq[::2]**2 - 1./fref**2)).to(u.s).value
    else:
        dt = (dispersion_delay_constant * dm *
                 (1./freq**2 - 1./fref**2)).to(u.s).value

    if dedisperse in ['coherent', 'by-channel']:
        # pre-calculate required turns due to dispersion
        if hasattr(fh, 'freqs'):
            fcoh = (freq[np.newaxis,:] +
                    fftfreq(ntint, dtsample.value)[:,np.newaxis] * u.Hz)
        else:
            if fedge_at_top:
                fcoh = fedge - thisfftfreq(nchan*2*ntint, dt1.value) * u.Hz
            else:
                fcoh = fedge + thisfftfreq(nchan*2*ntint, dt1.value) * u.Hz

        # set frequency relative to which dispersion is coherently corrected
        if dedisperse == 'coherent':
            _fref = fref
        else:
            # _fref = np.round((fcoh * dtsample).to(1).value) / dtsample
            _fref = np.repeat(freq.value, ntint) * freq.unit
        # (check via eq. 5.21 and following in
        # Lorimer & Kramer, Handbook of Pulsar Astrono
        dang = (dispersion_delay_constant * dm * fcoh *
                (1./_fref-1./fcoh)**2) * 360. * u.deg

        if fh.telescope == 'aro':
            # order of frequencies is r[0], r[1],i[1],...r[n-1],i[n-1],r[n]
            # for 0 and n need only real part, but for 1...n-1 need real, imag
            # so just get shifts for r[1], r[2], ..., r[n-1]
            dang = dang.to(u.rad).value[1:-1:2]
        else:
            dang = dang.to(u.rad).value
        dd_coh = np.exp(dang * 1j).conj().astype(np.complex64)


    for j in xrange(rank, nt, size):
        if verbose and j % progress_interval == 0:
            print('Doing {:6d}/{:6d}; time={:18.12f}'.format(
                j+1, nt, (tstart+dtsample*j*ntint).value))  # time since start

        # just in case numbers were set wrong -- break if file ends
        # better keep at least the work done
        try:
            # MPI processes read like a slinky
            if size > 1:
                fh.seek( (nskip+j) * count * itemsize)

            # ARO/GMRT return int-stream, LOFAR returns complex64 (count/nchan, nchan)
            raw = fh.record_read(count)

        except(EOFError, IOError) as exc:
            print("Hit {}; writing pgm's".format(exc))
            break
        if verbose >= 2:
            print("Read {} items".format(raw.size), end="")

        if rfi_filter_raw is not None:
            raw = rfi_filter_raw(raw)
            if verbose >= 2:
                print("... raw RFI", end="")

        if fh.telescope == 'lofar':
            vals = raw
        elif fh.telescope == 'aro':
            vals = raw.astype(np.float32)
        elif fh.telescope == 'gmrt':
            # data just a series of byte pairs, of real and imag
            vals = raw.astype(np.float32).view(np.complex64).squeeze()

        if dedisperse in ['coherent', 'by-channel']:
            fine = thisfft(vals, axis=0, overwrite_x=True, **_fftargs)
            if fh.telescope == 'lofar':
                fine *= dd_coh
            else:
                fine_cmplx = fine[1:-1].view(np.complex64)
                fine_cmplx *= dd_coh  # this overwrites parts of fine, as intended
            vals = thisifft(fine, axis=0, overwrite_x=True, **_fftargs)
            if verbose >= 2:
                print("... dedispersed", end="")

        if fh.telescope == 'lofar':
            power = vals.real**2 + vals.imag**2
        elif fh.telescope == 'aro':
            chan2 = thisfft(vals.reshape(-1, nchan*2), axis=-1,
                         overwrite_x=True, **_fftargs)**2
            # rfft: Re[0], Re[1], Im[1], ..., Re[n/2-1], Im[n/2-1], Re[n/2]
            # re-order to Num.Rec. format: Re[0], Re[n/2], Re[1], ....
            power = np.hstack((chan2[:,:1]+chan2[:,-1:],
                               chan2[:,1:-1].reshape(-1,nchan-1,2).sum(-1)))
        elif fh.telescope == 'gmrt':
            chan = vals.reshape(-1, nchan)
            power = chan.real**2+chan.imag**2

        if verbose >= 2:
            print("... power", end="")

        if rfi_filter_power is not None:
            power = rfi_filter_power(power)
            print("... power RFI", end="")

        # current sample positions in stream
        isr = j*ntint + np.arange(ntint)

        if do_waterfall:
            # loop over corresponding positions in waterfall
            for iw in xrange(isr[0]//ntw, isr[-1]//ntw + 1):
                if iw < nwsize:  # add sum of corresponding samples
                    waterfall[:,iw] += np.sum(power[isr//ntw == iw],
                                              axis=0)
            if verbose >=  2:
                print("... waterfall", end="")

        if do_foldspec:
            tsample = (tstart + isr*dtsample).value  # times since start
            ibin = j*ntbin//nt  # bin in the time series: 0..ntbin-1

            for k in xrange(nchan):
                if dedisperse == 'coherent':
                    t = tsample  # already dedispersed
                else:
                    t = tsample - dt[k]  # dedispersed times

                phase = phasepol(t)  # corresponding PSR phases
                iphase = np.remainder(phase*ngate,
                                      ngate).astype(np.int)
                # sum and count samples by phase bin
                foldspec[k, :, ibin] += np.bincount(iphase, power[:, k], ngate)
                icount[k, :, ibin] += np.bincount(iphase, power[:, k] != 0.,
                                                  ngate)

            if verbose >= 2:
                print("... folded", end="")

        if verbose >= 2:
            print("... done")

    if verbose:
        print('read {0:6d} out of {1:6d}'.format(j+1, nt))

    # update the 'subint' header
    fh['SUBINT'].header.update('NPOL', 1)
    fh['SUBINT'].header.update('NBIN', 1)
    fh['SUBINT'].header.update('NBIN_PRD', ngate)
    fh['SUBINT'].header.update('NCHAN', nchan)
    # AAR: entered random chan_bw... TODO
    fh['SUBINT'].header.update('CHAN_BW', 0.197)
    if dedisperse in ['coherent', 'by-channel', 'incoherent']:
        fh['SUBINT'].header.update('DM', dm)

    return foldspec, icount, waterfall


class Folder(dict):
    """
    convenience class to populate many of the 'fold' arguments
    from the psrfits headers of a datafile

    """
    def __init__(self, fh, **kwargs):

        # get the required arguments to 'fold'
        fold_args = getargspec(fold)
        fold_argnames = fold_args.args
        fold_defaults = fold_args.defaults
        # and set the defaults
        Nargs = len(fold_args.args)
        Ndefaults = len(fold_defaults)
        for i, v in enumerate(fold_defaults):
            self[fold_argnames[Nargs - Ndefaults + i]] = v
            
        # get some defaults from fh (may be overwritten by kwargs)
        self['dtype'] = fh.dtype
        self['samplerate'] = fh.samplerate #(1./fh['SUBINT'].header['TBIN']*u.Hz).to(u.MHz)
        self['fedge'] = fh.fedge
        self['fedge_at_top'] = fh.fedge_at_top
        self['nchan'] = fh['SUBINT'].header['NCHAN']
        self['ngate'] = fh['SUBINT'].header['NBIN_PRD']

        # update arguments with passed kwargs
        for k in kwargs:
            if k in fold_argnames:
                self[k] = kwargs[k]
            else:
                print("{} not needed for fold routine".format(k))
        # warn of missing, skipping fh and comm
        missing = [k for k in fold_argnames[2:] if k not in self]
        if len(missing) > 0:
            print("Missing 'fold' arguments: {}".format(missing))
                
    def __call__(self, fh, comm=None):
        ifold, icount, water = fold(fh, comm=comm, **self)
        return ifold, icount, water


def normalize_counts(q, count=None):
    """ normalize routines for waterfall and foldspec data """
    if count is None:
        nonzero = np.isclose(q, np.zeros_like(q)) # == 0.
        qn = q
    else:
        nonzero = count > 0
        qn = np.where(nonzero, q/count, 0.)
    qn -= np.where(nonzero,
                   np.sum(qn, 1, keepdims=True) /
                   np.sum(nonzero, 1, keepdims=True), 0.)
    return qn

