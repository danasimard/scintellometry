"""
load the observation data, which is stored as a ConfigObj object.

We do some parsing of the data in routine 'obsdata' to
get the data in a useful format

"""
from numpy.polynomial import Polynomial
from numpy import argmin
import re
from astropy import units as u
from astropy.coordinates import ICRS
from astropy.time import Time, TimeDelta

from astropy.extern.configobj_py2.configobj import ConfigObj
from astropy.utils.data import get_pkg_data_filename


def obsdata(conf='observations.conf'):
    """Load the observation data"""
    C = ConfigObj(get_pkg_data_filename(conf))

    # map things from ConfigObj to dictionary of useful objects
    obs = {}
    for key, val in C.iteritems():
        if key == 'psrs':
            obs[key] = parse_psrs(val)
        else:
            obs[key] = parse_tel(key,val)
    return obs


class telescope(dict):
    def __init__(self, name):
        assert name in ['aro', 'lofar', 'gmrt', 'arochime', 'jbdada']
        self['name'] = name
        self['observations'] = []

    def nearest_observation(self, t):
        """
        return key of nearest observation to (utc) time 't'.
        A warning is raised if the observation > 1s away

        """
        if isinstance(t, str):
            t = Time(t, scale='utc')

        dts = []
        dates = [self[d]['date'] for d in self['observations']]
        for date in dates:
            dts.append(abs((t - date).sec))
        dtmin = argmin(dts)
        key = self['observations'][dtmin]
        if dts[dtmin] > 1.:
            tmplt = ("Warning, nearest observation {0} is more than 1 second "
                     "away from request time {1}")
            print(tmplt.format(key, str(t)))
            # raise Warning(tmplt.format(key, str(t)))
        return key

    def file_list(self, key, **kwargs):
        """
        return list of files for observation 'key'
        The output depends on the telescope
        """
        seq = {'aro':self._aro_seq_raw_files,
               'lofar':self._lofar_file,
               'gmrt':self._gmrt_twofiles,
               'arochime':self._arochime_files,
               'jbdada':self._jbdada_files}
        return seq[self['name']](key, **kwargs)

    def _aro_seq_raw_files(self, key):
        """
        return the ARO sequence and raw files for observation 'key'

        """
        obs = self[key]
        fnbase = obs.get('fnbase', self.get('fnbase', None))
        disk_no = obs.get('disk_no', self.get('disk_no', None))
        node = obs.get('node', self.get('node', None))
        dt = key
        seq_file = (self['seq_filetmplt'].format(fnbase, disk_no[0], node, dt))
        raw_files = [self['raw_filestmplt'].format(fnbase, disk_no[i],
                                                   node, dt, i)
                     for i in range(3)]
        return (seq_file, raw_files)

    def _lofar_file(self, key):
        """
        return a list of 2-tuples for LOFAR observations 'key'.
        Each tuple is the S-set of files, and the list is over the
        P channels

        """
        obs = self[key]
        fnbase = obs.get('fnbase', self.get('fnbase', None))
        floc = obs.get('floc', None)
        file_fmt = self['file_fmt']
        S = obs.get('S', self.get('S', None))
        P = obs.get('P', self.get('P', None))
        files = []
        for p in P:
            subset = []
            for s in S:
                subset.append(file_fmt.format(fnbase, floc, S=s, P=p))
            files.append(subset)
        return (files,)  # protect from the *files done in GenericOpen

    def _gmrt_twofiles(self, key):
        """"
        return a 2-tuple for GMRT observation 'key':
        (timestamp file, [file1, file2])
        """
        obs = self[key]
        fnbase = obs.get('fnbase', self.get('fnbase', None))
        file_fmt = obs.get('file_fmt', self.get('file_fmt', None))
        pol = obs.get('pol', self.get('pol', None))
        file1 = file_fmt.format(fnbase, pol, 1)
        file2 = file_fmt.format(fnbase, pol, 2)
        timestamps = file1.split('.Pol')[0] + '.timestamp'
        return (timestamps, [file1, file2])

    def _arochime_files(self, key):
        """"
        return a 2-tuple for GMRT observation 'key':
        (timestamp file, [file1, file2])
        """
        obs = self[key]
        first = int(obs.get('first', self.get('first', 0)))
        last = int(obs.get('last', self.get('last', first)))
        fnbase = obs.get('fnbase', self.get('fnbase', None))
        file_fmt = obs.get('file_fmt', self.get('file_fmt', None))
        files = [file_fmt.format(fnbase, number)
                 for number in range(first, last+1)]
        return (files,)

    def _jbdada_files(self, key):
        """"
        return a 1-tuple for JB observation 'key':
        ([raw_files],)
        """
        obs = self[key]
        first = int(obs.get('first', self.get('first', 0)))
        last = int(obs.get('last', self.get('last', first)))
        filesize = int(obs.get('filesize', self.get('filesize', 640000000)))
        fnbase = obs.get('fnbase', self.get('fnbase', None))
        file_fmt = obs.get('file_fmt', self.get('file_fmt', None))
        files = [file_fmt.format(fnbase, key.replace('T', '-'), number)
                 for number in range(first, last+1, filesize)]
        return (files,)


class observation(dict):
    def __init__(self, date, val):
        self['date'] = date
        for k, v in val.iteritems():
            if k == 'ppol' and v.startswith('Polynomial'):
                self[k] = eval(v)
            elif k in ('P', 'S'):
                self[k] = [int(_v) for _v in v]
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

            polyco_file = get_pkg_data_filename(phasepol)
            # polyco = Polyco(polyco_file)
            # phasepol = polyco.phasepol(time0, rphase=rphase, t0=time0,
            #                            time_unit=u.second, convert=True)
            phasepol = PolycoPhasepol(polyco_file, time0, rphase=rphase,
                                      time_unit=time_unit, convert=convert)
        return phasepol


def parse_tel(telname, vals):
    tel = telescope(telname)
    for key, val in vals.iteritems():
        try:
            # then this is an observation
            date = Time(key, scale='utc')
            obs = observation(date, val)
            tel.update({key: obs})
            tel['observations'].append(key)
        except ValueError:
            if key in ('P', 'S'):
                val = [int(v) for v in val]
            tel.update({key: val})
    return tel


def parse_psrs(psrs):
    for name, vals in psrs.iteritems():
        if 'coords' not in vals:
            # add a coordinate attribute
            match = re.search("\d{4}[+-]\d+", name)
            if match is not None:
                crds = match.group()
                # set *very* rough position (assumes name format
                # [BJ]HHMM[+-]DD*)
                ra = '{0}:{1}'.format(crds[0:2], crds[2:4])
                dec = '{0}:{1}'.format(crds[5:7], crds[7:]).strip(':')
                vals['coords'] = ICRS(coordstr='{0}, {1}'.format(ra,dec),
                                      unit=(u.hour, u.degree))
            else:
                vals['coords'] = ICRS(coordstr='0, 0',
                                      unit=(u.hour, u.degree))
        else:
            coord = vals['coords']
            if coord.startswith("<ICRS RA"):
                # parse the (poor) ICRS print string
                ra = re.search('RA=[+-]?\d+\.\d+ deg', coord).group()
                dec = re.search('Dec=[+-]?\d+\.\d+ deg', coord).group()
                coord = '{0}, {1}'.format(ra[3:], dec[4:])
            vals['coords'] = ICRS(coordstr=coord)
        if 'dm' in vals:
            vals['dm'] = eval(vals['dm'])
    return psrs


if __name__ == '__main__':
    obsdata()
