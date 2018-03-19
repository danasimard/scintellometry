from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
import argparse

from astropy.time import Time, TimeDelta
import astropy.units as u

from scintellometry.folding.fold import Folder, normalize_counts
from scintellometry.folding.pmap import pmap

from .observations import obsdata

from mpi4py import MPI


def reduce(telescope, obskey, tstart, tend, nchan, ngate, ntbin, ntw_min,
           rfi_filter_raw=None, fref=None, dedisperse=None,
           rfi_filter_power=None, do_waterfall=True, do_foldspec=True,
           verbose=True):

    comm = MPI.COMM_WORLD
    if dedisperse == 'None':
        dedisperse = None

    if verbose > 3 and comm.rank == 0:
        print(telescope, obskey, tstart, tend, nchan, ngate, ntbin, ntw_min,
              rfi_filter_raw, fref, dedisperse,
              do_waterfall, do_foldspec, verbose)

    if dedisperse is not None and fref is None:
        raise ValueError("Need to give reference frequency to dedisperse to")

    obs = obsdata()
    if verbose > 3 and comm.rank == 0:
        print(obs)
    # find nearest observation to 'date',
    # warning if > 1s off requested start time of observation
    if obskey not in obs[telescope]:
        # assume it is a date, which may be only good to nearest second
        obskey = obs[telescope].nearest_observation(obskey)
    # target of this observation
    psr = obs[telescope][obskey]['src']
    assert psr in obs['psrs'].keys()

    dm = obs['psrs'][psr]['dm']

    if verbose and comm.rank == 0:
        print("Attempting to open {0}: {1}".format(telescope, obskey))

    with obs[telescope].open(obskey, comm=comm) as fh:
        if verbose and comm.rank == 0:
            print("Opened files")

        # nchan = None means data is channelized already, so we get this
        # property directly from the file
        if nchan is None:
            nchan = fh.nchan
        # ensure requested number of channels is integer multiple of
        # existing channels
        if nchan % getattr(fh, 'nchan', 1) != 0:
            raise ValueError("Can only channelize data to an integer multiple "
                             "of the number of input channels (={0})."
                             .format(fh.nchan))

        time0 = fh.time0
        tstart = time0 if tstart is None else Time(tstart, scale='utc')
        if tstart < time0:
            if time0 - tstart < 10*u.microsecond:
                print("Warning: Changed start time {0} to start of file {1}.".
                      format(tstart.iso,time0.iso))
                tstart = time0
            else:
                raise ValueError("Cowardly refusing to analyse with requested "
                             "time {0} before start time {1}."
                             .format(tstart.iso, time0.iso))
        if tend is None:
            tend = obs[telescope][obskey]['tend']

        try:
            tend = Time(tend, scale='utc')
            dt = tend - tstart
        except ValueError:
            dt = TimeDelta(float(tend), format='sec')
            tend = tstart + dt

        if verbose and comm.rank == 0:
            print("Requested time span: {0} to {1}".format(tstart.isot,
                                                           tend.isot))

        phasepol = obs[telescope][obskey].get_phasepol(time0)
        nt = fh.ntimebins(tstart, tend)
        ntint = fh.ntint(nchan if fh.nchan == 1 else fh.nchan)
        # number of samples to combine for waterfall
        ntw = min(ntw_min, nt*ntint)
        # number of records to skip

        if verbose and comm.rank == 0:
            print("Using start time {0} and phase polynomial {1}"
                  .format(time0.isot, phasepol))

        fh.seek(tstart)
        if verbose and comm.rank == 0:
            print("Skipped {0} blocks and will fold {1} blocks to cover "
                  "time span {2} to {3}"
                  .format(fh.offset/fh.blocksize, nt, fh.time().isot,
                          fh.time(fh.offset + nt*fh.blocksize).isot))

        # set the default parameters to fold
        # Note, some parameters may be in fh's HDUs, or fh.__getitem__
        # but these are overwritten if explicitly sprecified in Folder
        folder = Folder(fh, nchan=nchan,
                        nt=nt, ntint=ntint, ngate=ngate,
                        ntbin=ntbin, ntw=ntw, dm=dm, fref=fref,
                        phasepol=phasepol, dedisperse=dedisperse,
                        do_waterfall=do_waterfall, do_foldspec=do_foldspec,
                        verbose=verbose, progress_interval=1,
                        rfi_filter_raw=rfi_filter_raw,
                        rfi_filter_power=rfi_filter_power)
        myfoldspec, myicount, mywaterfall = folder(fh, comm=comm)
    # end with

    print("Rank {0} exited with statement".format(comm.rank))

    savepref = "{0}{1}_{2}chan{3}ntbin".format(telescope, psr, nchan, ntbin)
    if do_waterfall:
        waterfall = np.zeros_like(mywaterfall)
        comm.Reduce(mywaterfall, waterfall, op=MPI.SUM, root=0)
        if comm.rank == 0:
            # waterfall = normalize_counts(waterfall)
            np.save("{0}waterfall_{1}+{2:08}sec.npy"
                    .format(savepref, tstart.isot, dt.sec), waterfall)

    if do_foldspec:
        foldspec = np.zeros_like(myfoldspec) if comm.rank == 0 else None
        print("Rank {0} is entering comm.Reduce".format(comm.rank))
        comm.Reduce(myfoldspec, foldspec, op=MPI.SUM, root=0)
        del myfoldspec  # save memory on node 0
        icount = np.zeros_like(myicount) if comm.rank == 0 else None
        comm.Reduce(myicount, icount, op=MPI.SUM, root=0)
        del myicount  # save memory on node 0
        if comm.rank == 0:
            fname = ("{0}foldspec_{1}+{2:08}sec.npy")
            np.save(fname.format(savepref, tstart.isot, dt.sec), foldspec)
            iname = ("{0}icount_{1}+{2:08}sec.npy")
            np.save(iname.format(savepref, tstart.isot, dt.sec), icount)

    if comm.rank == 0:
        if do_foldspec and foldspec.ndim == 3:
            # sum over time slices -> pulse profile vs channel
            foldspec1 = normalize_counts(foldspec.sum(0).astype(np.float64),
                                         icount.sum(0).astype(np.int64))
            # sum over channels -> pulse profile vs time
            foldspec3 = normalize_counts(foldspec.sum(1).astype(np.float64),
                                         icount.sum(1).astype(np.int64))
            fluxes = foldspec1.sum(axis=0)
            with open('{0}flux_{1}+{2:08}sec.dat'
                      .format(savepref, tstart.isot, dt.sec), 'w') as f:
                for i, flux in enumerate(fluxes):
                    f.write('{0:12d} {1:12.9g}\n'.format(i+1, flux))
            # ratio'd flux only if file will not be ridiculously large
            if ntbin*ngate < 10000:
                foldspec2 = normalize_counts(foldspec, icount)

        plots = True
        if plots and do_waterfall and waterfall.ndim == 2:  # no polarizations
            w = waterfall.copy()
            pmap('{0}waterfall_{1}+{2:08}sec.pgm'
                 .format(savepref, tstart.isot, dt.sec), w, 1, verbose=True)
        if plots and do_foldspec and foldspec.ndim == 3:
            pmap('{0}folded_{1}+{2:08}sec.pgm'
                 .format(savepref, tstart.isot, dt.sec), foldspec1, 0, verbose)
            pmap('{0}folded3_{1}+{2:08}sec.pgm'
                 .format(savepref, tstart.isot, dt.sec),
                 foldspec3.T, 0, verbose)
            # ratio'd flux only if file will not be ridiculously large
            if ntbin*ngate < 10000:
                pmap('{0}foldedbin_{1}+{2:08}sec.pgm'
                     .format(savepref, tstart.isot, dt.sec),
                     foldspec2.transpose(1,2,0).reshape(nchan,-1), 1, verbose)

    # savefits = False
    # if savefits and comm.rank == 0:
    #     print("Saving FITS files...")
    #     # assign the folded data ( [f2] = [nchan, ngate, ntbin]
    #     #                          want [ntbin, npol, nchan, ngate]

    #     # TODO: figure out lofar and aro data, which concatenates channels
    #     # so f2 = len(range(P))*nchan
    #     if telescope == 'lofar':
    #         f2 = f2[-nchan:]
    #     elif telescope == 'aro':
    #         f2 = f2[0:nchan]
    #     nchan, ngate, ntbin = f2.shape
    #     f2 = f2.transpose(2, 0, 1)
    #     f2 = f2.reshape(ntbin, np.newaxis, nchan, ngate)
    #     std = f2.std(axis=0)
    #     subint_table[1].data.field('DATA')[:] = (2**16*f2/std).astype(
    #         np.int16)
    #     fout = ('{0}folded3_{1}+{2:08}sec.fits'
    #             .format(savepref, tstart.isot, dt.sec))
    #     # Note: output_verify != 'ignore' resets the cards for some reason
    #     try:
    #         subint_table.writeto(fout, output_verify='ignore', clobber=True)
    #     except ValueError:
    #         print("FITS writings is a work in progress: "
    #               "need to come to terms with array shapes")
    #         print("f2.shape, subint_table[1].data.field('DATA')[:].shape = ",
    #               f2.shape, subint_table[1].data.field('DATA')[:].shape)


def CL_parser():
    parser = argparse.ArgumentParser(
        prog='reduce_data.py',
        description='Process data, with options to dedisperse and fold.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--reduction_defaults', type=str,
        help="One of ['aro', 'lofar', 'gmrt', 'arochime'].\n"
        "A convenience flag to set default parameters as defined at the "
        "end of reduce_data.py (TO DO: make dicts)")

    d_parser = parser.add_argument_group(
        "Data-related parameters \n"
        "They specify which observation run to process "
        "(consistent with [telescope], [[date]] entries\n"
        "in observations.conf), and the start and finish timestamps.")
    d_parser.add_argument(
        '-t','--telescope', type=str, default='gmrt',
        help="The data to reduce. One of ['gmrt', 'kairo', 'lofar', 'arochime'].")
    d_parser.add_argument(
        '-d','--date','--observation', type=str, default='2014-01-19T22:22:48',
        help="The date or other identifier of the data to reduce. "
        "The key with which the observation is stored in observations.conf.")
    d_parser.add_argument(
        '-t0', '--tstart', type=str, default=None,
        help="Timestamp within the observation run to start processing."
        "Default is start of the observation")
    d_parser.add_argument(
        '-t1', '--tend', '-dt', '--duration', type=str, default=None,
        help="Timestamp within the observation run to end processing or "
        "duration (in seconds) of the section to process.")
    d_parser.add_argument(
        '--rfi_filter_raw', action='store_true',
        help="Apply the 'rfi_filter_rwa' routine to the raw data.")
    d_parser.add_argument(
        '--rfi_filter_power', action='store_true',
        help="Apply the 'rfi_filter_power' routine to "
              "possibly dedispersed spectra.")

    f_parser = parser.add_argument_group("folding related parameters")
    f_parser.add_argument(
        '-f','--foldspec', type=bool, default=True,
        help="Produce a folded spectrum")
    f_parser.add_argument(
        '-nc', '--nchan', type=int, default=None,
        help="Number of channels in folded spectrum.")
    f_parser.add_argument(
        '-ng', '--ngate', type=int, default=256,
        help="number of bins over the pulsar period.")
    f_parser.add_argument(
        '-nb', '--ntbin', type=int, default=5,
        help="number of time bins the time series is split into for folding.")

    w_parser = parser.add_argument_group("Waterfall related parameters")
    w_parser.add_argument(
        '-w','--waterfall', type=bool, default=False,
        help="Produce a waterfall plot")
    w_parser.add_argument(
        '-nwm', '--ntw_min', type=int, default=10200,
        help="number of samples to combine for waterfall")

    d_parser = parser.add_argument_group("Dedispersion related parameters.")
    d_parser.add_argument(
        '--dedisperse', type=str, default='incoherent',
        help="One of ['None', 'coherent', 'by-channel', 'incoherent'].\n"
        "Note: None really does nothing.")
    d_parser.add_argument(
        '--fref', type=float, default=None,
        help="reference frequency for dispersion measure")

    parser.add_argument('-v', '--verbose', action='append_const', const=1)
    return parser.parse_args()
