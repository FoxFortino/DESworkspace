import astropy.coordinates as co

class Gnomonic(object):
    ''' Class representing a gnomonic projection about some point on
    the sky.  Can be used to go between xi,eta coordinates and ra,dec.
    All xy units are assumed to be in degrees as are the ra, dec, and PA of
    the projection pole.  Uses astropy.coordinates.
    '''
    @staticmethod
    def type():
        return 'Gnomonic'

    def __init__(self, ra, dec, rotation=0.):
        '''
        Create a Gnomonic transformation by specifying the position of the
        pole (in ICRS degrees) and rotation angle of the axes relative
        to ICRS north.

        :param ra,dec: ICRS RA and Declination of the pole of the projection.
        :param rotation: position angle (in degrees) of the projection axes.
        '''
        self.pole_ra = ra
        self.pole_dec = dec
        self.rotation = rotation
        self.frame = None

    def _set_frame(self):
        pole = co.SkyCoord(self.pole_ra, self.pole_dec, unit='deg',frame='icrs')
        self.frame = pole.skyoffset_frame(rotation=co.Angle(self.rotation,unit='deg'))

    def toSky(self, x, y):
        '''
        Convert xy coordinates in the gnomonic project (in degrees) into ra, dec.
        '''
        try:
            import coord
            pole = coord.CelestialCoord(self.pole_ra * coord.degrees,
                                        self.pole_dec * coord.degrees)
            deg_per_radian = coord.radians / coord.degrees
            # Coord wants these in radians, not degrees
            # Also, a - sign for x, since astropy uses +ra as +x direction.
            x /= -deg_per_radian
            y /= deg_per_radian
            # apply rotation
            if self.rotation != 0.:
                # TODO: I'm not sure if I have the sense of the rotation correct here.
                #       The "complex wcs" test has PA = 0, so I wasn't able to test it.
                #       There may be a sign error on the s terms.
                s, c = (self.rotation * coord.degrees).sincos()
                x, y = x*c - y*s, x*s + y*c
            # apply projection
            ra, dec = pole.deproject_rad(x, y, projection='gnomonic')
            ra *= deg_per_radian
            dec *= deg_per_radian
            return ra, dec

        except ImportError:
            if self.frame is None: self._set_frame()

            # Get the y and z components of unit-sphere coords, x on pole axis
            y, z = x, y
            y *= np.pi / 180.
            z *= np.pi / 180.
            temp = np.sqrt(1 + y*y + z*z)
            y /= temp
            z /= temp
            dec = np.arcsin(z)
            ra = np.arcsin(y / np.cos(dec))
            coord = co.SkyCoord(ra, dec, unit='rad', frame=self.frame)
            return coord.icrs.ra.deg, coord.icrs.dec.deg

    def toXY(self, ra, dec):
        '''
        Convert RA, Dec into xy values in the gnomonic projection, in degrees
        '''
        if self.frame is None: self._set_frame()
            
        coord = co.SkyCoord(ra, dec, unit='deg')
        s = coord.transform_to(self.frame)
        
        # Get 3 components on unit sphere
        x = np.cos(s.lat.radian)*np.cos(s.lon.radian)
        y = np.cos(s.lat.radian)*np.sin(s.lon.radian)
        z = np.sin(s.lat.radian)
        out_x = y/x * (180. / np.pi)
        out_y = z/x * (180. / np.pi)
        return out_x, out_y