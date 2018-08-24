"""Classes for handling base-band recorded data in a consistent way,
so that routines downstream can read data without having to worry about
the way it is stored on disks.  The classes are used as a base to handle, e.g.,
ARO, LOFAR, and GMRT data in the respective modules."""

from __future__ import division

import numpy as np
import os
import warnings

from astropy import units as u
from astropy.time import Time
from .fromfile import fromfile
try:
    from mpi4py import MPI
except ImportError:
    pass
from .psrfits_tools import psrFITS


# size in bytes of records read from file (simple for ARO: 1 byte/sample)
def dtype_itemsize(dtype):
    bps = {'ci1': 2, '(2,)ci1': 4, 'ci1,ci1': 4, '1bit': 0.125, 
           '2bit': 0.25, '4bit': 0.5, '4bit,4bit': 1, 'c4bit': 1, 
           'cu4bit,cu4bit': 2}.get(dtype, None)
    if bps is None:
        bps = np.dtype(dtype).itemsize
    return bps

# default properties for various telescopes
header_defaults = {}


class MultiFile(psrFITS):

    def __init__(self, files=None, blocksize=None, dtype=None, nchan=None,
                 comm=None):
        if comm is None:
            try:
                self.comm = MPI.COMM_SELF
            except:
                self.comm = None
        else:
            self.comm = comm
        # parameters for fold:
        if blocksize is not None:
            self.blocksize = blocksize
        if dtype is not None:
            self.dtype = dtype
            self.data_is_complex = dtype[:1] == 'c'
        if nchan is not None:
            self.nchan = nchan
        self.itemsize = dtype_itemsize(self.dtype)
        self.recordsize = self.itemsize * self.nchan
        assert self.blocksize % self.recordsize == 0
        self.setsize = int(self.blocksize // self.recordsize)

        super(MultiFile, self).__init__(hdus=['SUBINT'])
        #self.set_hdu_defaults(header_defaults[self.telescope])
        if files is not None:
            self.open(files)

    def set_hdu_defaults(self, dictionary):
        for hdu in dictionary:
            self[hdu].header.update(dictionary[hdu])

    def open(self, files):
        self.fh_raw = [open(raw, 'rb') for raw in files]
        self.offset = 0

    def close(self):
        for fh in self.fh_raw:
            fh.close()

    def read(self, size):
        """Read size bytes, returning an ndarray with np.int8 dtype.

        Incorporate information from multiple underlying files if necessary.
        The individual file pointers are assumed to be pointing at the right
        locations, i.e., just before data that will be read here.
        """
        if size % self.recordsize != 0:
            raise ValueError("Cannot read a non-integer number of records")

        # ensure we do not read beyond end
        size = min(size, len(self.indices) * self.blocksize - self.offset)
        if size <= 0:
            raise EOFError('At end of file in MultiFile.read')

        # allocate buffer for MPI read
        z = np.empty(size, dtype=np.int8)

        # read one or more pieces
        iz = 0
        while(iz < size):
            block, already_read = divmod(self.offset, self.blocksize)
            fh_size = min(size - iz, self.blocksize - already_read)
            fh_index = self.indices[block]
            if fh_index >= 0:
                z[iz:iz+fh_size] = np.fromstring(self.fh_raw[fh_index]
                                                 .read(fh_size), dtype=z.dtype)
            else:
                z[iz:iz+fh_size] = 0
            self.offset += fh_size
            iz += fh_size

        return z

    def seek(self, offset):
        """Move filepointers to given offset

        Parameters
        ----------
        offset : float, Quantity, TimeDelta, Time, or str (iso-t)
            If float, in units of bytes
            If Quantity in time units or TimeDelta, interpreted as offset from
                start time, and converted to nearest record
            If Time, calculate offset from start time and convert
        """
        if isinstance(offset, Time):
            offset = offset-self.time0
        elif isinstance(offset, str):
            offset = Time(offset, scale='utc') - self.time0

        try:
            offset = offset.to(self.dtsample.unit)
        except AttributeError:
            pass
        except u.UnitsError:
            offset = int(offset.to(u.byte).value)
        else:
            offset = (offset/self.dtsample).to(u.dimensionless_unscaled).value
            offset = int(round(offset) * self.recordsize)
        self._seek(offset)

    def _seek(self, offset):
        if offset % self.recordsize != 0:
            raise ValueError("Cannot offset to non-integer number of records")
        # determine index in units of the blocksize
        block, extra = divmod(offset, self.blocksize)
        if block > len(self.indices):
            raise EOFError('At end of file in MultiFile.read')

        # check how many of the indices preceding the block were in each file
        indices = self.indices[:block]
        fh_offsets = np.bincount(indices[indices >= 0],
                                 minlength=len(self.fh_raw)) * self.blocksize
        # add the extra bytes to the correct file
        if self.indices[block] >= 0:
            fh_offsets[self.indices[block]] += extra

        # actual seek in files
        for fh, fh_offset in zip(self.fh_raw, fh_offsets):
            fh.seek(fh_offset)

        self.offset = offset

    def tell(self, offset=None, unit=None):
        if offset is None:
            offset = self.offset

        if unit is None:
            return offset

        if isinstance(unit, str) and unit == 'time':
            return self.time(offset)

        return (offset * u.byte).to(
            unit, equivalencies=[(u.Unit(self.recordsize * u.byte),
                                  u.Unit(self.dtsample))])

    def time(self, offset=None):
        """Get time corresponding to the current (or given) offset"""
        if offset is None:
            offset = self.offset
        if offset % self.recordsize != 0:
            warnings.warn("Offset for which time is requested is not "
                          "integer multiple of record size.")
        return self.time0 + self.tell(offset, u.day)

    # ARO and GMRT (LOFAR_Pcombined overwrites this)
    def seek_record_read(self, offset, count):
        """Read count samples starting from offset (also in samples)"""
        self.seek(offset)
        return self.record_read(count)

    def record_read(self, count):
        return fromfile(self, self.dtype,
                        count).reshape(-1, self.nchan).squeeze()

    def nskip(self, date, time0=None):
        """
        Return the number of records needed to skip from start of
        file to iso timestamp 'date'.

        Optionally:
        time0 : use this start time instead of self.time0
                either a astropy.time.Time object or string in 'utc'
        """
        time0 = self.time0 if time0 is None else Time(time0, scale='utc')
        dt = Time(date, scale='utc') - time0
        nskip = int(round((dt / self.dtsample / self.setsize)
                          .to(u.dimensionless_unscaled)))
        return nskip

    def ntint(self, nchan):
        return (self.setsize * self.nchan // nchan //
                (1 if self.data_is_complex else 2))

    def ntimebins(self, t0, t1):
        """
        determine the number of timebins between UTC start time 't0'
        and end time 't1'
        """
        t0 = Time(t0, scale='utc')
        t1 = Time(t1, scale='utc')
        nt = ((t1-t0).to(u.s) / self.dtsample /
              (self.setsize)).to(u.dimensionless_unscaled).value
        return np.round(nt).astype(int)

    # for use in context manager ("with <MultiFile> as fh:")
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class SequentialFile(MultiFile):
    """Class for readers that read from data stored in a sequence of files."""
    def __init__(self, raw_files, *args, **kwargs):
        self.files = raw_files
        self.filesize = os.path.getsize(self.files[0])
        super(SequentialFile, self).__init__(None, *args, **kwargs)
        self.current_file_number = None
        self.header_size = 0
        self.open(0)
        self.offset = 0

    def open(self, number=0):
        """Open a new file in the sequence.

        Parameters
        ----------
        file_number : int
            The number of the file to open.  Default is 0, i.e., the first one.
        """
        if number != self.current_file_number:
            self.close()
            self.fh_raw = open(self.files[number], mode='rb')
            self.current_file_number = number
        return self.fh_raw

    def close(self):
        """Close the current raw file."""
        if self.current_file_number is not None:
            self.fh_raw.close()
            self.current_file_number = None

    def _seek(self, offset):
        """Skip to given offset, possibly opening a new file."""
        assert offset % self.recordsize == 0
        file_number, file_offset = divmod(offset,
                                          self.filesize - self.header_size)
        self.open(file_number)
        self.fh_raw.seek(file_offset + self.header_size)
        self.offset = offset

    def read(self, size):
        """Read size bytes, returning an ndarray with np.int8 dtype.

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
        z = np.empty(size, dtype=np.int8)

        # read one or more pieces
        iz = 0
        while(iz < size):
            self._seek(self.offset)
            block, already_read = divmod(self.offset, self.filesize)
            fh_size = min(size - iz, self.filesize - already_read)
            z[iz:iz+fh_size] = np.fromstring(self.fh_raw.read(fh_size),
                                             dtype=z.dtype)
            iz += fh_size
            self.offset += fh_size

        return z

    def __repr__(self):
        if self.current_file_number is not None:
            return ("<SequentialFile: file {0} open, offset {1} (time {2})>"
                    .format(self.files[self.current_file_number],
                            self.offset, self.time().isot))
        else:
            return ("<SequentialFile: no files open, file list={0}>"
                    .format(self.files))

# Don't need MPI.File for reading, but may need it for writing; would require
# something like this in `open`.
#        # MPI.File.Open doesn't handle files with ":"
#        self.fh_raw = []
#        self.fh_links = []
#        for raw in files:
#            fname, islnk = good_name(os.path.abspath(raw))
#            self.fh_raw.append(MPI.File.Open(self.comm, fname,
#                                             amode=MPI.MODE_RDONLY))
#            if islnk:
#                self.fh_links.append(fname)
