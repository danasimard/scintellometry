"""
Load the observation data, which is stored as a ConfigObj object.

We do some parsing of the data in routine 'obsdata' to get the data
in a more useful format.
"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import warnings

import numpy as np
from numpy.polynomial import Polynomial
import re
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

from astropy.extern.configobj import configobj
from collections import OrderedDict

from scintellometry import io


__all__ = ['obsdata']


DATA_READERS = {reader.telescope: reader
                for reader in io.__dict__.values()
                if callable(reader) and hasattr(reader, 'telescope')}


def aro_seq_raw_files(seq_filetmplt, raw_filestmplt, fnbase,
                      key, disk_no, node, **kwargs):
    """
    return the ARO sequence and raw files for observation 'key'.
    """
    seq_file = seq_filetmplt.format(fnbase, disk_no[0], node, key)
    raw_files = [raw_filestmplt.format(fnbase, disk_no[i], node, key, i)
                 for i in range(3)]
    return (seq_file, raw_files)


def lofar_file(file_fmt, fnbase, floc, S, P, **kwargs):
    """
    return a list of 2-tuples for LOFAR observations 'key'.
    Each tuple is the S-set of files, and the list is over the
    P channels
    """
    files = []
    for p in P:
        subset = []
        for s in S:
            subset.append(file_fmt.format(fnbase, floc, S=s, P=p))
        files.append(subset)
    return (files,)  # protect from the *files done in GenericOpen


def gmrt_rawfiles(file_fmt, fnbase, **kwargs):
    """"
    return a 2-tuple for GMRT observation 'key':
    (timestamp file, [file1, ...])

    Number of files depends on whether there is a ``nodes`` keyword argument
    """
    nodes = kwargs.get('nodes', None)
    if nodes is None:
        files = [file_fmt.format(fnbase)]
    else:
        files = [file_fmt.format(fnbase, node) for node in nodes]
    timestamps = files[0] + '.timestamp'
    return (timestamps, files)


def gmrt_twofiles(file_fmt, fnbase, pol, **kwargs):
    """"
    return a 2-tuple for GMRT observation 'key':
    (timestamp file, [file1, file2])
    """
    file1 = file_fmt.format(fnbase, pol, 1)
    file2 = file_fmt.format(fnbase, pol, 2)
    timestamps = file1.split('.Pol')[0] + '.timestamp'
    return (timestamps, [file1, file2])


def vlbi_files(file_fmt, fnbase, first=0, **kwargs):
    """"
    return a 1-tuple for VLBI-like observations, which contains just
    a list of files: ([files],)
    """
    last = kwargs.get('last', first)
    files = [file_fmt.format(fnbase, number)
             for number in range(int(first), int(last)+1)]
    return (files,)


def dada_files(file_fmt, fnbase, key, first, **kwargs):
    """"
    return a 1-tuple for JB observation 'key':
    ([raw_files],)
    """
    last = kwargs.get('last', first)
    filesize = kwargs.get('filesize', 640000000)
    files = [file_fmt.format(fnbase, key.replace('T', '-'), number)
             for number in range(int(first), int(last)+1, int(filesize))]
    return (files,)


FILE_LIST_PICKERS = {
    'aro': aro_seq_raw_files,
    'lofar': lofar_file,
    'gmrt': gmrt_twofiles,
    'gmrt-raw': gmrt_rawfiles,
    'arochime': vlbi_files,
    'arochime-raw': vlbi_files,
    'arochime-vdif': vlbi_files,
    'arochime-invpfb': vlbi_files,
    'arochime-raw-invpfb': vlbi_files,
    'dada': dada_files,
    'jbdada': dada_files,
    'mark4': vlbi_files,
    'mark5b': vlbi_files,
    'vdif': vlbi_files}


class Telescope(OrderedDict):
    def __init__(self, name):
        super(Telescope, self).__init__(name=name, observations=[])

    def nearest_observation(self, t):
        """
        return key of nearest observation to (utc) time 't'.
        A warning is raised if the observation > 2s away
        """
        if isinstance(t, str):
            t = Time(t, scale='utc')

        dts = np.array([abs((self[d]['date'] - t).sec)
                        for d in self['observations']])
        dtmin = dts.argmin()
        key = self['observations'][dtmin]
        if dts[dtmin] > 2.:
            warnings.warn("Warning, nearest observation {0} is more than "
                          "2 seconds away from request time {1}."
                          .format(key, str(t)))
        return key

    def open(self, key, comm=None):
        """Open the reader with the files associated with `key`."""
        data_format = self.get('format', self['name'])
        if data_format not in DATA_READERS:
            raise ValueError("Unsupported data format {0}".format(data_format))
        setup = self.get('setup', {})
        setup.update(self[key].get('setup', {}))
        file_setup = {'key': key}
        file_setup.update(self)
        file_setup.update(self[key])
        files = FILE_LIST_PICKERS[data_format](**file_setup)
        return DATA_READERS[data_format](*files, comm=comm, **setup)


class Observation(dict):
    def __init__(self, date, val):
        self['date'] = date
        for k, v in val.iteritems():
            if k == 'ppol' and v.startswith('Polynomial'):
                self[k] = eval(v)
            elif k in ('P', 'S', 'channels', 'nodes'):
                self[k] = [int(_v) for _v in v]
            elif k == 'setup':
                self[k] = parse_setup(v)
            else:
                self[k] = v

    def get_phasepol(self, time0, rphase='fraction', time_unit=u.second,
                     convert=True):
        """
        return the phase polynomial at time0
        (calculated if necessary)
        """
        phasepol = self['ppol']
        if phasepol is None:
            subs = [self['src'], str(self['date'])]
            wrn = "{0} is not configured for time {1} \n".format(*subs)
            wrn += "\tPlease update observations.conf "
            raise Warning(wrn)

        elif not isinstance(phasepol, Polynomial):
            # Assume phasepol holds a file containing polycos.
            from pulsar.predictor import Polyco

            class PolycoPhasepol(object):
                """Polyco wrapper that will get phase relative to some
                reference time0, picking the appropriate polyco chunk."""
                def __init__(self, polyco_file, time0, rphase, time_unit,
                             convert):
                    self.polyco = Polyco(polyco_file)
                    self.time0 = time0
                    self.rphase = rphase
                    self.time_unit = time_unit
                    self.convert = convert

                def __call__(self, dt):
                    """Get phases for time differences dt (float in seconds)
                    relative to self.time0 (filled by initialiser).

                    Chunks are assumed to be sufficiently closely spaced that
                    one can get the index into the polyco table from the
                    first item.
                    """
                    try:
                        dt0 = dt[0]
                    except IndexError:
                        dt0 = dt

                    time0 = self.time0 + TimeDelta(dt0, format='sec')
                    phasepol = self.polyco.phasepol(
                        time0, rphase=self.rphase, t0=time0,
                        time_unit=self.time_unit, convert=self.convert)
                    return phasepol(dt-dt0)

            phasepol = PolycoPhasepol(phasepol, time0, rphase=rphase,
                                      time_unit=time_unit, convert=convert)
        return phasepol


def obsdata(conf='observations.conf'):
    """Load the observation data."""
    # C = configobj.ConfigObj(get_pkg_data_filename(conf))
    C = configobj.ConfigObj(conf)

    # map things from ConfigObj to dictionary of useful objects
    obs = {}
    for key, val in C.iteritems():
        if key == 'psrs' or key == 'pulsars':
            obs[key] = parse_pulsars(val)
        else:
            obs[key] = parse_telescope(key, val)
    return obs


def parse_telescope(name, vals):
    tel = Telescope(name)
    for key, val in vals.iteritems():
        try:
            # if parsable as a Time, this key describes an observation.
            date = Time(key, scale='utc')
            tel['observations'].append(key)
            val = Observation(date, val)
        except ValueError:
            if key in ('P', 'S', 'channels', 'nodes'):
                val = [int(v) for v in val]
            elif key == 'setup':
                val = parse_setup(val)
        tel[key] = val
    return tel


def parse_pulsars(psrs):
    for name, vals in psrs.iteritems():
        if 'coords' not in vals:
            # add a coordinate attribute
            match = re.search("\d{4}[+-]\d+", name)
            if match is not None:
                crds = match.group()
                # set *very* rough position (assumes name format
                # [BJ]HHMM[+-]DD*)
                ra = '{0}:{1}'.format(crds[0:2], crds[2:4])
                dec = '{0}:{1}'.format(crds[4:7], crds[7:]).strip(':')
                vals['coords'] = SkyCoord('{0} {1}'.format(ra, dec),
                                          unit=(u.hour, u.degree))
            else:
                vals['coords'] = SkyCoord('0 0', unit=(u.hour, u.degree))
        else:
            coord = vals['coords']
            if coord.startswith("<ICRS RA"):
                # parse the (poor) ICRS print string
                ra = re.search('RA=[+-]?\d+\.\d+ deg', coord).group()
                dec = re.search('Dec=[+-]?\d+\.\d+ deg', coord).group()
                coord = '{0} {1}'.format(ra[3:], dec[4:])
            vals['coords'] = SkyCoord(coord)

        if 'dm' in vals:
            vals['dm'] = eval(vals['dm'])

    return psrs


def parse_setup(setup):
    for k, v in setup.iteritems():
        if k in ('P', 'S', 'channels', 'nodes'):
            setup[k] = [int(_v) for _v in v]
        else:
            setup[k] = eval(v)
    return setup


if __name__ == '__main__':
    obsdata()
