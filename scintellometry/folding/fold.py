from __future__ import division, print_function

from inspect import getargspec
import numpy as np
import os
import astropy.units as u


try:
    # do *NOT* use on-disk cache; blue gene doesn't work; slower anyway
    # import pyfftw
    # pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.numpy_fft import (rfft, irfft,
                                             fft, ifft, fftfreq)
    from numpy.fft import rfftfreq  # Missing from pyfftw for some reason.
    # By default, scipy_fftpack just uses scipy.rfftfreq, but this is
    # inconsistent with the order actually in the arrays, which is that
    # also used by numpy.
    _fftargs = {'threads': int(os.environ.get('OMP_NUM_THREADS', 2)),
                'planner_effort': 'FFTW_ESTIMATE',
                'overwrite_input': True}
    _rfftargs = _fftargs
except(ImportError):
    print("Consider installing pyfftw: https://github.com/hgomersall/pyFFTW")
    # Use complex FFT from scipy, since unlike numpy it does not cast up to
    # complex128.  However, use rfft from numpy, since the scipy data order
    # is too tricky to use.
    from scipy.fftpack import fft, ifft, fftfreq
    from numpy.fft import rfft, irfft, rfftfreq

    _fftargs = {'overwrite_x': True}
    _rfftargs = {}

dispersion_delay_constant = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc


def fold(fh, comm, samplerate, fedge, fedge_at_top, nchan,
         nt, ntint, ngate, ntbin, ntw, dm, fref, phasepol,
         dedisperse='incoherent',
         do_waterfall=True, do_foldspec=True, verbose=True,
         progress_interval=100, rfi_filter_raw=None, rfi_filter_power=None,
         return_fits=False):
    """
    FFT data, fold by phase/time and make a waterfall series

    Folding is done from the position the file is currently in

    Parameters
    ----------
    fh : file handle
        handle to file holding voltage timeseries
    comm: MPI communicator or None
        will use size, rank attributes
    samplerate : Quantity
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
        start of the file that is read.
    dedisperse : None or string (default: incoherent).
        None, 'incoherent', 'coherent', 'by-channel'.
        Note: None really does nothing
    do_waterfall, do_foldspec : bool
        whether to construct waterfall, folded spectrum (default: True)
    verbose : bool or int
        whether to give some progress information (default: True)
    progress_interval : int
        Ping every progress_interval sets
    return_fits : bool (default: False)
        return a subint fits table for rank == 0 (None otherwise)

    """
    assert dedisperse in (None, 'incoherent', 'by-channel', 'coherent')
    need_fine_channels = dedisperse in ['by-channel', 'coherent']
    assert nchan % fh.nchan == 0
    if dedisperse in ['incoherent', 'by-channel'] and fh.nchan > 1:
        oversample = nchan // fh.nchan
        assert ntint % oversample == 0
    else:
        oversample = 1

    if dedisperse == 'coherent' and fh.nchan > 1:
        raise ValueError("Cannot coherently dedisperse channelized data.")

    if comm is None:
        mpi_rank = 0
        mpi_size = 1
    else:
        mpi_rank = comm.rank
        mpi_size = comm.size

    npol = getattr(fh, 'npol', 1)
    assert npol == 1 or npol == 2
    if verbose > 1 and mpi_rank == 0:
        print("Number of polarisations={}".format(npol))

    # initialize folded spectrum and waterfall
    # TODO: use estimated number of points to set dtype
    if do_foldspec:
        foldspec = np.zeros((ntbin, nchan, ngate, npol**2), dtype=np.float32)
        icount = np.zeros((ntbin, nchan, ngate), dtype=np.int32)
    else:
        foldspec = None
        icount = None

    if do_waterfall:
        nwsize = nt*ntint//ntw//oversample
        waterfall = np.zeros((nwsize, nchan, npol**2), dtype=np.float64)
    else:
        waterfall = None

    if verbose and mpi_rank == 0:
        print('Reading from {}'.format(fh))

    nskip = fh.tell()/fh.blocksize
    if nskip > 0:
        if verbose and mpi_rank == 0:
            print('Starting {0} blocks = {1} bytes out from start.'
                  .format(nskip, nskip*fh.blocksize))

    dt1 = (1./samplerate).to(u.s)
    # need 2*nchan real-valued samples for each FFT
    if fh.telescope == 'lofar':
        dtsample = fh.dtsample
    else:
        dtsample = nchan // oversample * 2 * dt1
    tstart = dtsample * ntint * nskip

    # pre-calculate time delay due to dispersion in coarse channels
    # for channelized data, frequencies are known

    tb = -1. if fedge_at_top else +1.
    if fh.nchan == 1:
        if getattr(fh, 'data_is_complex', False):
            # for complex data, really each complex sample consists of
            # 2 real ones, so multiply dt1 by 2.
            freq = fedge + tb * fftfreq(nchan, 2.*dt1)
            if dedisperse == 'coherent':
                fcoh = fedge + tb * fftfreq(nchan*ntint, 2.*dt1)
                fcoh.shape = (-1, 1)
            elif dedisperse == 'by-channel':
                fcoh = freq + tb * fftfreq(ntint, dtsample)[:, np.newaxis]
        else:  # real data
            freq = fedge + tb * rfftfreq(nchan*2, dt1)
            if dedisperse == 'coherent':
                fcoh = fedge + tb * rfftfreq(ntint*nchan*2, dt1)
                fcoh.shape = (-1, 1)
            elif dedisperse == 'by-channel':
                fcoh = freq + tb * fftfreq(ntint, dtsample)[:, np.newaxis]
        freq_in = freq

    else:
        # Input frequencies may not be the ones going out.
        freq_in = fh.frequencies
        if oversample == 1:
            freq = freq_in
        else:
            freq = freq_in[:, np.newaxis] + tb * fftfreq(oversample, dtsample)

        fcoh = freq_in + tb * fftfreq(ntint, dtsample)[:, np.newaxis]

    # print('fedge_at_top={0}, tb={1}'.format(fedge_at_top, tb))
    # By taking only up to nchan, we remove the top channel at the Nyquist
    # frequency for real, unchannelized data.
    ifreq = freq[:nchan].ravel().argsort()

    # pre-calculate time offsets in (input) channelized streams
    dt = dispersion_delay_constant * dm * (1./freq_in**2 - 1./fref**2)

    if need_fine_channels:
        # pre-calculate required turns due to dispersion.
        #
        # set frequency relative to which dispersion is coherently corrected
        if dedisperse == 'coherent':
            _fref = fref
        else:
            _fref = freq_in[np.newaxis, :]
        # (check via eq. 5.21 and following in
        # Lorimer & Kramer, Handbook of Pulsar Astronomy
        dang = (dispersion_delay_constant * dm * fcoh *
                (1./_fref-1./fcoh)**2) * u.cycle
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            dd_coh = np.exp(dang * 1j).conj().astype(np.complex64)

        # add dimension for polarisation
        dd_coh = dd_coh[..., np.newaxis]

    # Calculate the part of the whole file this node should handle.
    size_per_node = (nt-1)//mpi_size + 1
    start_block = mpi_rank*size_per_node
    end_block = min((mpi_rank+1)*size_per_node, nt)
    for j in range(start_block, end_block):
        if verbose and j % progress_interval == 0:
            print('#{:4d}/{:4d} is doing {:6d}/{:6d} [={:6d}/{:6d}]; '
                  'time={:18.12f}'
                  .format(mpi_rank, mpi_size, j+1, nt,
                          j-start_block+1, end_block-start_block,
                          (tstart+dtsample*j*ntint).value))  # time since start

        # Just in case numbers were set wrong -- break if file ends;
        # better keep at least the work done.
        try:
            raw = fh.seek_record_read(int((nskip+j)*fh.blocksize),
                                      fh.blocksize)
        except(EOFError, IOError) as exc:
            print("Hit {0!r}; writing data collected.".format(exc))
            break
        if verbose >= 2:
            print("#{:4d}/{:4d} read {} items"
                  .format(mpi_rank, mpi_size, raw.size), end="")

        if npol == 2 and raw.dtype.fields is not None:
            raw = raw.view(raw.dtype.fields.values()[0][0])

        if fh.nchan == 1:  # raw.shape=(ntint*npol)
            raw = raw.reshape(-1, npol)
        else:              # raw.shape=(ntint, nchan*npol)
            raw = raw.reshape(-1, fh.nchan, npol)

        if dedisperse == 'incoherent' and oversample > 1:
            raw = ifft(raw, axis=1, **_fftargs).reshape(-1, nchan, npol)
            raw = fft(raw, axis=1, **_fftargs)

        if rfi_filter_raw is not None:
            raw, ok = rfi_filter_raw(raw)
            if verbose >= 2:
                print("... raw RFI (zap {0}/{1})"
                      .format(np.count_nonzero(~ok), ok.size), end="")

        if np.can_cast(raw.dtype, np.float32):
            vals = raw.astype(np.float32)
        else:
            assert raw.dtype.kind == 'c'
            vals = raw

        # For pre-channelized data, data are always complex,
        # and should have shape (ntint, nchan, npol).
        # For baseband data, we wish to get to the same shape for
        # incoherent or by_channel, or just to fully channelized for coherent.
        if fh.nchan == 1:
            # If we need coherent dedispersion, do FT of whole thing,
            # otherwise to output channels, mimicking pre-channelized data.
            if raw.dtype.kind == 'c':  # complex data
                nsamp = len(vals) if dedisperse == 'coherent' else nchan
                vals = fft(vals.reshape(-1, nsamp, npol), axis=1,
                           **_fftargs)
            else:  # real data
                nsamp = len(vals) if dedisperse == 'coherent' else nchan * 2
                vals = rfft(vals.reshape(-1, nsamp, npol), axis=1,
                            **_rfftargs)
                # Sadly, the way data are stored depends on what FFT routine
                # one is using.  We cannot deal with scipy's.
                if vals.dtype.kind == 'f':
                    raise TypeError("Can no longer deal with scipy's format "
                                    "for storing FTs of real data.")

        if fedge_at_top:
            # take complex conjugate to ensure by-channel de-dispersion is
            # applied correctly.
            # This needs to be done for ARO data, since we are in 2nd Nyquist
            # zone; not clear it is needed for other telescopes.
            np.conj(vals, out=vals)

        # Now we coherently dedisperse, either all of it or by channel.
        if need_fine_channels:
            # for by_channel, we have vals.shape=(ntint, nchan, npol),
            # and want to FT over ntint to get fine channels;
            if vals.shape[0] > 1:
                fine = fft(vals, axis=0, **_fftargs)
            else:
                # for coherent, we just reshape:
                # (1, ntint*nchan, npol) -> (ntint*nchan, 1, npol)
                fine = vals.reshape(-1, 1, npol)

            # Dedisperse.
            fine *= dd_coh

            # Still have fine.shape=(ntint, nchan, npol),
            # w/ nchan=1 for coherent.
            if fine.shape[1] > 1 or raw.dtype.kind == 'c':
                vals = ifft(fine, axis=0, **_fftargs)
            else:
                vals = irfft(fine, axis=0, **_rfftargs)

            if fine.shape[1] == 1 and nchan > 1:
                # final FT to get requested channels
                if vals.dtype.kind == 'f':
                    vals = vals.reshape(-1, nchan*2, npol)
                    vals = rfft(vals, axis=1, **_rfftargs)
                else:
                    vals = vals.reshape(-1, nchan, npol)
                    vals = fft(vals, axis=1, **_fftargs)
            elif dedisperse == 'by-channel' and oversample > 1:
                vals = vals.reshape(-1, oversample, fh.nchan, npol)
                vals = fft(vals, axis=1, **_fftargs)
                vals = vals.transpose(0, 2, 1, 3).reshape(-1, nchan, npol)

            # vals[time, chan, pol]
            if verbose >= 2:
                print("... dedispersed", end="")

        if npol == 1:
            power = vals.real**2 + vals.imag**2
        else:
            p0 = vals[..., 0]
            p1 = vals[..., 1]
            power = np.empty(vals.shape[:-1] + (4,), np.float32)
            power[..., 0] = p0.real**2 + p0.imag**2
            power[..., 1] = p0.real*p1.real + p0.imag*p1.imag
            power[..., 2] = p0.imag*p1.real - p0.real*p1.imag
            power[..., 3] = p1.real**2 + p1.imag**2

        if verbose >= 2:
            print("... power", end="")

        # current sample positions and corresponding time in stream
        isr = j*(ntint // oversample) + np.arange(ntint // oversample)
        tsr = (isr*dtsample*oversample)[:, np.newaxis]

        if rfi_filter_power is not None:
            power = rfi_filter_power(power, tsr.squeeze())
            print("... power RFI", end="")

        # correct for delay if needed
        if dedisperse in ['incoherent', 'by-channel']:
            # tsample.shape=(ntint/oversample, nchan_in)
            tsr = tsr - dt

        if do_waterfall:
            # # loop over corresponding positions in waterfall
            # for iw in range(isr[0]//ntw, isr[-1]//ntw + 1):
            #     if iw < nwsize:  # add sum of corresponding samples
            #         waterfall[iw, :] += np.sum(power[isr//ntw == iw],
            #                                    axis=0)[ifreq]
            iw = np.round((tsr / dtsample / oversample).to(1)
                          .value / ntw).astype(int)
            for k, kfreq in enumerate(ifreq):  # sort in frequency while at it
                iwk = iw[:, (0 if iw.shape[1] == 1 else kfreq // oversample)]
                iwk = np.clip(iwk, 0, nwsize-1, out=iwk)
                iwkmin = iwk.min()
                iwkmax = iwk.max()+1
                for ipow in range(npol**2):
                    waterfall[iwkmin:iwkmax, k, ipow] += np.bincount(
                        iwk-iwkmin, power[:, kfreq, ipow], iwkmax-iwkmin)
            if verbose >= 2:
                print("... waterfall", end="")

        if do_foldspec:
            ibin = (j*ntbin) // nt  # bin in the time series: 0..ntbin-1

            # times and cycles since start time of observation.
            tsample = tstart + tsr
            phase = (phasepol(tsample.to(u.s).value.ravel())
                     .reshape(tsample.shape))
            # corresponding PSR phases
            iphase = np.remainder(phase*ngate, ngate).astype(np.int)

            for k, kfreq in enumerate(ifreq):  # sort in frequency while at it
                iph = iphase[:, (0 if iphase.shape[1] == 1
                                 else kfreq // oversample)]
                # sum and count samples by phase bin
                for ipow in range(npol**2):
                    foldspec[ibin, k, :, ipow] += np.bincount(
                        iph, power[:, kfreq, ipow], ngate)
                icount[ibin, k, :] += np.bincount(
                    iph, power[:, kfreq, 0] != 0., ngate).astype(np.int32)

            if verbose >= 2:
                print("... folded", end="")

        if verbose >= 2:
            print("... done")

    #Commented out as workaround, this was causing "Referenced before assignment" errors with JB data
    #if verbose >= 2 or verbose and mpi_rank == 0:
    #    print('#{:4d}/{:4d} read {:6d} out of {:6d}'
    #          .format(mpi_rank, mpi_size, j+1, nt))

    if npol == 1:
        if do_foldspec:
            foldspec = foldspec.reshape(foldspec.shape[:-1])
        if do_waterfall:
            waterfall = waterfall.reshape(waterfall.shape[:-1])

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
        self['samplerate'] = fh.samplerate
        # ??? (1./fh['SUBINT'].header['TBIN']*u.Hz).to(u.MHz)
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
        return fold(fh, comm=comm, **self)


def normalize_counts(q, count=None):
    """ normalize routines for waterfall and foldspec data """
    if count is None:
        nonzero = np.isclose(q, np.zeros_like(q))  # == 0.
        qn = q
    else:
        nonzero = count > 0
        qn = np.where(nonzero, q/count, 0.)
    # subtract mean profile (pulsar phase always last dimension)
    qn -= np.where(nonzero,
                   np.sum(qn, -1, keepdims=True) /
                   np.sum(nonzero, -1, keepdims=True), 0.)
    return qn

# if return_fits and mpi_rank == 0:
#     # subintu HDU
#     # update table columns
#     # TODO: allow multiple polarizations
#     npol = 1
#     newcols = []
#     # FITS table creation difficulties...
#     # assign data *after* 'new_table' creation
#     array2assign = {}
#     tsubint = ntint*dtsample
#     for col in fh['subint'].columns:
#         attrs = col.copy().__dict__
#         # remove non-init args
#         for nn in ['_pseudo_unsigned_ints', '_dims', '_physical_values',
#                    'dtype', '_phantom', 'array']:
#             attrs.pop(nn, None)

#         if col.name == 'TSUBINT':
#             array2assign[col.name] = np.array(tsubint)
#         elif col.name == 'OFFS_SUB':
#             array2assign[col.name] = np.arange(ntbin) * tsubint
#         elif col.name == 'DAT_FREQ':
#             # TODO: sort from lowest freq. to highest
#             # ('DATA') needs sorting as well
#             array2assign[col.name] = freq.to(u.MHz).value.astype(np.double)
#             attrs['format'] = '{0}D'.format(freq.size)
#         elif col.name == 'DAT_WTS':
#             array2assign[col.name] = np.ones(freq.size, dtype=np.float32)
#             attrs['format'] = '{0}E'.format(freq.size)
#         elif col.name == 'DAT_OFFS':
#             array2assign[col.name] = np.zeros(freq.size*npol,
#                                               dtype=np.float32)
#             attrs['format'] = '{0}E'.format(freq.size*npol)
#         elif col.name == 'DAT_SCL':
#             array2assign[col.name] = np.ones(freq.size*npol,
#                                              dtype=np.float32)
#             attrs['format'] = '{0}E'.format(freq.size)
#         elif col.name == 'DATA':
#             array2assign[col.name] = np.zeros((ntbin, npol, freq.size,
#                                                ngate), dtype='i1')
#             attrs['dim'] = "({},{},{})".format(ngate, freq.size, npol)
#             attrs['format'] = "{0}I".format(ngate*freq.size*npol)
#         newcols.append(FITS.Column(**attrs))
#     newcoldefs = FITS.ColDefs(newcols)

#     oheader = fh['SUBINT'].header.copy()
#     newtable = FITS.new_table(newcoldefs, nrows=ntbin, header=oheader)
#     # update the 'subint' header and create a new one to be returned
#     # owing to the structure of the code (MPI), we need to assign
#     # the 'DATA' outside of fold.py
#     newtable.header.update('NPOL', 1)
#     newtable.header.update('NBIN', ngate)
#     newtable.header.update('NBIN_PRD', ngate)
#     newtable.header.update('NCHAN', freq.size)
#     newtable.header.update('INT_UNIT', 'PHS')
#     newtable.header.update('TBIN', tsubint.to(u.s).value)
#     chan_bw = np.abs(np.diff(freq.to(u.MHz).value).mean())
#     newtable.header.update('CHAN_BW', chan_bw)
#     if dedisperse in ['coherent', 'by-channel', 'incoherent']:
#         newtable.header.update('DM', dm.value)
#     # finally assign the table data
#     for name, array in array2assign.iteritems():
#         try:
#             newtable.data.field(name)[:] = array
#         except ValueError:
#             print("FITS error... work in progress",
#                   name, array.shape, newtable.data.field(name)[:].shape)

#     phdu = fh['PRIMARY'].copy()
#     subinttable = FITS.HDUList([phdu, newtable])
#     subinttable[1].header.update('EXTNAME', 'SUBINT')
#     subinttable['PRIMARY'].header.update('DATE-OBS', fh.time0.isot)
#     subinttable['PRIMARY'].header.update('STT_IMJD', int(fh.time0.mjd))
#     subinttable['PRIMARY'].header.update(
#         'STT_SMJD', int(str(fh.time0.mjd - int(fh.time0.mjd))[2:])*86400)

# return subinttable
