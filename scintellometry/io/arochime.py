#           _____   ____   _____ _    _ _____ __  __ ______
#     /\   |  __ \ / __ \ / ____| |  | |_   _|  \/  |  ____|
#    /  \  | |__) | |  | | |    | |__| | | | | \  / | |__
#   / /\ \ |  _  /| |  | | |    |  __  | | | | |\/| |  __|
#  / ____ \| | \ \| |__| | |____| |  | |_| |_| |  | | |____
# /_/    \_\_|  \_\\____/ \_____|_|  |_|_____|_|  |_|______|

from __future__ import division
import io

import numpy as np
from scipy.fftpack import fftfreq, fftshift
from astropy.time import Time
import astropy.units as u

from . import SequentialFile, header_defaults

import pfb

class AROCHIMEData(SequentialFile):

    telescope = 'arochime'

    def __init__(self, raw_files, blocksize, samplerate, fedge, fedge_at_top,
                 time_offset=0.0*u.s, dtype='cu4bit,cu4bit', comm=None):
        """ARO data aqcuired with a CHIME correlator containts 1024 channels
        over the 400MHz BW, 2 polarizations, and 2 unsigned 8-byte ints for
        real and imaginary for each timestamp.
        """
        self.meta = eval(open(raw_files[0] + '.meta').read())
        nchan = self.meta['nfreq']
        self.time0 = Time(self.meta['stime'], format='unix') + time_offset
        self.npol = self.meta['ninput']
        self.samplerate = samplerate
        self.fedge_at_top = fedge_at_top
        if fedge.isscalar:
            self.fedge = fedge
            f = fftshift(fftfreq(nchan, (2./samplerate).to(u.s).value)) * u.Hz
            if fedge_at_top:
                self.frequencies = fedge - (f-f[0])
            else:
                self.frequencies = fedge + (f-f[0])
        else:
            assert fedge.shape == (nchan,)
            self.frequencies = fedge
            if fedge_at_top:
                self.fedge = self.frequencies.max()
            else:
                self.fedge = self.frequencies.min()

        self.dtsample = (nchan * 2 / samplerate).to(u.s)
        if comm is None or comm.rank == 0:
            print("In AROCHIMEData, calling super")
            print("Start time: ", self.time0.iso)

        super(AROCHIMEData, self).__init__(raw_files, blocksize, dtype, nchan,
                                           comm=comm)


class ARORawFile(object):
    def __init__(self, fn, header_size, data_size):
        self.fh_raw = open(fn, mode='rb')
        self.offset = 0
        self.header_size = header_size
        self.data_size = data_size
        self.packet_size = header_size + data_size

    def seek(self, offset):
        self.offset = offset

    def tell(self):
        return self.offset

    def read(self, size):
        z = io.BytesIO()
        iz = 0
        while iz < size:
            block, already_read = divmod(self.offset, self.data_size)
            self.fh_raw.seek(block * self.packet_size + self.header_size +
                             already_read)
            fh_size = min(size - iz, self.data_size - already_read)
            z.write(self.fh_raw.read(fh_size))
            self.seek(self.offset + fh_size)
            iz += fh_size
        z.seek(0)
        return z.read()

    def close(self):
        return self.fh_raw.close()


header_dtype = np.dtype([('valid', '<u4'),
                         ('unused_header', '26b'),
                         ('n_frames', '<u4'),
                         ('n_input', '<u4'),
                         ('n_freq', '<u4'),
                         ('offset_freq', '<u4'),
                         ('seconds', '<u4'),
                         ('micro_seconds', '<u4'),
                         ('seq', '<u4')])


class AROCHIMERawData(SequentialFile):

    telescope = 'arochime-raw'

    def __init__(self, raw_files, blocksize, samplerate, fedge, fedge_at_top,
                 time_offset=0.0*u.s, dtype='cu4bit,cu4bit', comm=None):
        """ARO data acquired with a CHIME correlator, but not passed through
        the decode_stream script.

        Header has 58 bytes, data typically 4*1024*2=8192, hence total is
        8250.  Files seem to typically have 8192 of these packets.
        """
        header = np.fromfile(raw_files[0], dtype=header_dtype, count=1)[0]
        assert header['valid']
        self.time0 = Time(header['seconds'] + 1.e-6 * header['micro_seconds'],
                          format='unix', scale='utc')
        self.npol = header['n_input']
        nchan = header['n_freq']
        self.samplerate = samplerate
        self.fedge_at_top = fedge_at_top
        if fedge.isscalar:
            self.fedge = fedge
            f = fftshift(fftfreq(nchan, (2./samplerate).to(u.s).value)) * u.Hz
            if fedge_at_top:
                self.frequencies = fedge - (f-f[0])
            else:
                self.frequencies = fedge + (f-f[0])
        else:
            assert fedge.shape == (nchan,)
            self.frequencies = fedge
            if fedge_at_top:
                self.fedge = self.frequencies.max()
            else:
                self.fedge = self.frequencies.min()

        self.dtsample = (nchan * 2 / samplerate).to(u.s)
        if comm is None or comm.rank == 0:
            print("In AROCHIMERawData, calling super")
            print("Start time: ", self.time0.iso)

        self.data_size = (header['n_frames'] * header['n_freq'] *
                          header['n_input'])
        super(AROCHIMERawData, self).__init__(raw_files, blocksize, dtype,
                                              nchan, comm=comm)
        if self.filesize % (header_dtype.itemsize + self.data_size) != 0:
            raise ValueError("File size is not an integer number of packets")
        # self['SUBINT'].header.update(header)  # header is not a dict
        #
        # fake a filesize that would be correct without headers
        self.filesize = (self.filesize * self.data_size //
                         (self.data_size + header_dtype.itemsize))

    def open(self, number=0):
        """Open a new file in the sequence.

        Parameters
        ----------
        file_number : int
            The number of the file to open.  Default is 0, i.e., the first one.
        """
        if number != self.current_file_number:
            self.close()
            self.fh_raw = ARORawFile(self.files[number],
                                     header_dtype.itemsize, self.data_size)
            self.current_file_number = number
        return self.fh_raw


class AROCHIMEInvPFB(SequentialFile):

    telescope = 'arochime-invpfb'

    def __init__(self, raw_files, blocksize, samplerate, fedge, fedge_at_top,
                 time_offset=0.0*u.s, dtype='4bit,4bit', comm=None):
        """ARO data acquired with a CHIME correlator, PFB inverted.
        """
        self.fh_raw = AROCHIMERawData(raw_files, blocksize, samplerate,
                                      fedge, fedge_at_top, time_offset,
                                      dtype='cu4bit,cu4bit', comm=comm)
        self.time0 = self.fh_raw.time0
        self.samplerate = self.fh_raw.samplerate
        self.dtsample = (1 / self.samplerate).to(u.s)
        self.npol = self.fh_raw.npol
        self.fedge = self.fh_raw.fedge
        self.fedge_at_top = self.fh_raw.fedge_at_top
        super(AROCHIMEInvPFB, self).__init__(raw_files, blocksize, dtype,
                                             nchan=1, comm=comm)
        # PFB information
        self.nblock = 2048
        self.h = pfb.sinc_hamming(4, self.nblock).reshape(4, -1)
        self.fh = None
        # S/N for use in the Wiener Filter
        prec = (1 << 3) ** 0.5
        self.sn = prec / 0.2887
        print("Opened InvPFB reader, S/N={0}".format(self.sn))

    def open(self, number=0):
        self.fh_raw.open(number)

    def close(self):
        self.fh_raw.close()

    def _seek(self, offset):
        if offset % self.recordsize != 0:
            raise ValueError("Cannot offset to non-integer number of records")
        self.offset = (offset // self.fh_raw.recordsize *
                       self.fh_raw.recordsize)

    def seek_record_read(self, offset, size):
        print('Inverting PFB... ')
        raw_start = (offset // self.fh_raw.recordsize) * self.fh_raw.recordsize
        raw_end = ((offset + size + self.fh_raw.recordsize - 1) //
                   self.fh_raw.recordsize) * self.fh_raw.recordsize
        raw = self.fh_raw.seek_record_read(raw_start, raw_end-raw_start)
        if self.npol == 2:
            raw = raw.view(raw.dtype.fields.values()[0][0])

        raw = raw.reshape(-1, self.fh_raw.nchan, self.npol)

        nyq_pad = np.zeros((raw.shape[0], 1, self.npol), dtype=raw.dtype)
        raw = np.concatenate((raw, nyq_pad), axis=1)
        # Get pseudo-timestream
        print('Getting pseudo timestream')
        pd = np.fft.irfft(raw.conj(), axis=1)
        # Set up for deconvolution
        print('Setting up for deconvolution')
        fpd = np.fft.rfft(pd, axis=0)
        del pd
        if self.fh is None or self.fh.shape[0] != fpd.shape[0]:
            print('Getting FT of PFB')
            lh = np.zeros((raw.shape[0], self.h.shape[1]))
            lh[:self.h.shape[0]] = self.h
            self.fh = np.fft.rfft(lh, axis=0).conj()
        # FT of Wiener deconvolution kernel
        fg = self.fh.conj() / (np.abs(self.fh)**2 + (1/self.sn)**2)
        # Deconvolve and get deconvolved timestream
        print('Wiener deconvolving')
        rd = np.fft.irfft(fpd * fg[..., np.newaxis],
                          axis=0).reshape(-1, self.npol)
        # select actual part requested
        print('Selecting and returning')
        rd = rd[offset - raw_start:offset - raw_start + size]
        self.offset = offset + size
        # view as a record array
        return rd.astype('f4').view('f4,f4')


# GMRT defaults for psrfits HDUs
# Note: these are largely made-up at this point
header_defaults['arochime'] = {
    'PRIMARY': {'TELESCOP':'AROCHIME',
                'IBEAM':1, 'FD_POLN':'LIN',
                'OBS_MODE':'SEARCH',
                'ANT_X':0, 'ANT_Y':0, 'ANT_Z':0, 'NRCVR':1,
                'FD_HAND':1, 'FD_SANG':0, 'FD_XYPH':0,
                'BE_PHASE':0, 'BE_DCC':0, 'BE_DELAY':0,
                'TRK_MODE':'TRACK',
                'TCYCLE':0, 'OBSFREQ':300, 'OBSBW':100,
                'OBSNCHAN':0, 'CHAN_DM':0,
                'EQUINOX':2000.0, 'BMAJ':1, 'BMIN':1, 'BPA':0,
                'SCANLEN':1, 'FA_REQ':0,
                'CAL_FREQ':0, 'CAL_DCYC':0, 'CAL_PHS':0, 'CAL_NPHS':0,
                'STT_IMJD':54000, 'STT_SMJD':0, 'STT_OFFS':0},
    'SUBINT': {'INT_TYPE': 'TIME',
               'SCALE': 'FluxDen',
               'POL_TYPE': 'AABB',
               'NPOL':1,
               'NBIN':1, 'NBIN_PRD':1,
               'PHS_OFFS':0,
               'NBITS':1,
               'ZERO_OFF':0, 'SIGNINT':0,
               'NSUBOFFS':0,
               'NCHAN':1,
               'CHAN_BW':1,
               'DM':0, 'RM':0, 'NCHNOFFS':0,
               'NSBLK':1}}


header_defaults['arochime-raw'] = header_defaults['arochime']
header_defaults['arochime-invpfb'] = header_defaults['arochime']

def read_start_time(filename):
    """
    Reads in arochime .meta file as a dictionary and gets start time.

    Parameters
    ----------
    filename: str
         full path to .meta file

    Returns
    -------
    start_time: Time object
         Unix time of observation start
    """
    f = eval(open(filename).read())
    return Time(f['stime'], format='unix')
