#           _____   ____   _____ _    _ _____ __  __ ______
#     /\   |  __ \ / __ \ / ____| |  | |_   _|  \/  |  ____|
#    /  \  | |__) | |  | | |    | |__| | | | | \  / | |__
#   / /\ \ |  _  /| |  | | |    |  __  | | | | |\/| |  __|
#  / ____ \| | \ \| |__| | |____| |  | |_| |_| |  | | |____
# /_/    \_\_|  \_\\____/ \_____|_|  |_|_____|_|  |_|______|

from __future__ import division
import io
import os

import numpy as np
from numpy.fft import fftfreq, fftshift
from astropy.time import Time
import astropy.units as u

from . import SequentialFile, header_defaults

from ..ppf import pfb
from baseband import vdif

try:
    from pyfftw.interfaces.numpy_fft import rfft, irfft
    _fftargs = {'threads': int(os.environ.get('OMP_NUM_THREADS', 2)),
                'planner_effort': 'FFTW_ESTIMATE',
                'overwrite_input': True}
    _rfftargs = _fftargs
except:
    from numpy.fft import rfft, irfft
    print('pyfftw not installed, resorting to numpy fft. This will be less memory efficient')
    _fftargs = {'overwrite_x': True}
    _rfftargs = {}

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

class AROCHIMEVdifData(SequentialFile):

    telescope = 'arochime-vdif'

    def __init__(self, raw_files, blocksize, samplerate, fedge, fedge_at_top,
                 time_offset=0.0*u.s, dtype='cu4bit,cu4bit', comm=None):
        """ARO data acquired with a CHIME correlator, saved in VDIF format.
        Files are 2**16 time, 2 pol, 1024 freq, at 800MHz / (2*1024) samplerate
        Read with baseband VDIF package, need byte to sample conversions for folding
        """

        fh = vdif.open(raw_files[0], 'rs', sample_rate=samplerate)
        self.time0 = fh.tell(unit='time')
        self.npol = fh.nthread
        nchan = fh.nchan
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
            print("In AROCHIMEVdifData, calling super")
            print("Start time: ", self.time0.iso)

        super(AROCHIMEVdifData, self).__init__(raw_files, blocksize, dtype, nchan,
                                           comm=comm)
        if self.filesize % self.fh_raw.header0.framesize != 0:
            raise ValueError("File size is not an integer number of packets")

        self.filesize = (self.filesize // self.fh_raw.header0.framesize *
                         self.fh_raw.header0.payloadsize)

    def open(self, number=0):
        """Open a new file in the sequence.

        Parameters
        ----------
        file_number : int
            The number of the file to open.  Default is 0, i.e., the first one.
        """
        if number != self.current_file_number:
            self.close()
            self.fh_raw = vdif.open(self.files[number], 'rs', sample_rate=(1/self.dtsample).to(u.Hz))
            self.current_file_number = number

        return self.fh_raw

    def record_read(self, count):
        return self.read(count)

    def _seek(self, offset):
        """Skip to given offset, possibly opening a new file."""
        assert offset % self.recordsize == 0
        file_number, file_offset = divmod(offset,self.filesize)
        self.open(file_number)
        self.fh_raw.seek(file_offset//self.recordsize)
        self.offset = offset

    def read(self, size):
        """Read size bytes, returning an ndarray as np.float32 or np.complex64.

        Incorporate information from multiple underlying files if necessary.
        The current file pointer are assumed to be pointing at the right
        locations, i.e., just before the first bit of data that will be read.
        """
        if size % self.recordsize != 0:
            raise ValueError("Cannot read a non-integer number of records")

        # ensure we do not read beyond end
        size = min(size, len(self.files) * self.filesize - self.offset)
        if size <= 0:
            raise EOFError('At end of file!')

        # allocate buffer.
        z = np.empty((size//self.recordsize, self.nchan, self.npol),
                     dtype=np.complex64 if self.data_is_complex else np.float32)

        # read one or more pieces
        iz = 0
        while(iz < size):
            self._seek(self.offset)
            block, already_read = divmod(self.offset, self.filesize)
            fh_size = int(min(size - iz, self.filesize - already_read))
            z[iz//self.recordsize:(iz+fh_size)//self.recordsize] = self.fh_raw.read(fh_size // self.recordsize).transpose(0, 2, 1)
            iz += fh_size
            self.offset += fh_size

        return z

class AROCHIMEInvPFB(SequentialFile):
    telescope = 'arochime-invpfb'
    _raw_data_class = AROCHIMEVdifData

    def __init__(self, raw_files, blocksize, samplerate, fedge, fedge_at_top,
                 time_offset=0.0*u.s, dtype='4bit,4bit', comm=None):
        """ARO data acquired with a CHIME correlator, PFB inverted.

        The PFB inversion is imperfect at the edges. To do this properly,
        need to read in multiple blocks at a time (not currently implemented).

        Also, this will ideally be read as sets of 2048 samples
        (ie: read as dtype (2048,)4bit: 1024)
        """

        self.fh_raw = self._raw_data_class(raw_files, blocksize, samplerate,
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
        # Assume 8 bits are set to have noise at 3 bits, so 1.5 bits for FT.
        # samples off by uniform distribution of [-0.5, 0.5] ->
        # equivalent to adding noise with std=0.2887
        prec = (1 << 3) ** 0.5
        self.sn = prec / 0.2887

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
        if offset % self.recordsize != 0 or size % self.recordsize != 0:
            raise ValueError("size and offset must be an integer number of records")

        raw = self.fh_raw.seek_record_read(offset, size)

        if self.npol == 2 and self._raw_data_class == AROCHIMERawData:
            raw = raw.view(raw.dtype.fields.values()[0][0])

        raw = raw.reshape(-1, self.fh_raw.nchan, self.npol)
        nyq_pad = np.zeros((raw.shape[0], 1, self.npol), dtype=raw.dtype)
        raw = np.concatenate((raw, nyq_pad), axis=1)
        # Get pseudo-timestream
        pd = irfft(raw, axis=1, **_rfftargs)
        # Set up for deconvolution
        fpd = rfft(pd, axis=0, **_rfftargs)
        del pd
        if self.fh is None or self.fh.shape[0] != fpd.shape[0]:
            lh = np.zeros((raw.shape[0], self.h.shape[1]))
            lh[:self.h.shape[0]] = self.h
            self.fh = rfft(lh, axis=0, **_rfftargs).conj()
            del lh
        # FT of Wiener deconvolution kernel
        fg = self.fh.conj() / (np.abs(self.fh)**2 + (1/self.sn)**2)
        # Deconvolve and get deconvolved timestream
        rd = irfft(fpd * fg[..., np.newaxis],
                          axis=0, **_rfftargs).reshape(-1, self.npol)
        # select actual part requested
        self.offset = offset + size
        # view as a record array
        return rd.astype('f4')

class AROCHIMERawInvPFB(AROCHIMEInvPFB):

    telescope = 'arochime-raw-invpfb'
    _raw_data_class = AROCHIMERawData

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
header_defaults['arochime-raw-invpfb'] = header_defaults['arochime']
header_defaults['arochime-vdif'] = header_defaults['arochime']

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
