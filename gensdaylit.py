"""
gensdaylit, is a Python-based alternative to the classic Radiance
sky generator gendaylit, developed to model both the spectral composition
of daylight and the angular extent of the solar disc. 

"""


import os, math
import numpy as np
import pandas as pd
import posixpath
from pathlib import Path
from typing import Tuple, List, Optional
import textwrap
from loguru import logger
import pvlib

def write_rad_file(filepath: str, text: str, mute: bool = False) -> None:
    """Write a Radiance `.rad` text description to disk.

    Ensures `filepath` (as a `pathlib.Path`) is written with `text`. Logs
    success unless `mute` is True, and raises on failure.

    Args:
        filepath: Path to the `.rad` file to create.
        text: The full Radiance description to write.
        mute: If False, logs an INFO message upon successful write.

    Raises:
        IOError: If the file cannot be written.
    """
    filepath = Path(filepath)  # ensure its a Path
    try:
        filepath.write_text(text)
        if not mute:
            logger.info(f"Written RAD file: {filepath}")
    except Exception as exc:
        logger.error(f"Cannot write RAD file at {filepath}: {exc}")
        raise

def make_path(base: str, *segments: str) -> str:
    """Join base path with one or more subdirectories or filenames."""
    return posixpath.join(base, *segments)

def rotation_matrix(axis, theta_degrees):
    theta = np.radians(theta_degrees)
    if axis == 'x':
        return np.array([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

# -------------------------------------------------------------------
# Module-level constants
# -------------------------------------------------------------------
# Perez coefficients a-e (Table 1) as a function of sky clearness
_PEREZ_COEFF = [
    1.3525, -0.2576, -0.2690, -1.4366, -0.7670, 0.0007, 1.2734, -0.1233, 2.8000, 0.6004,
    1.2375, 1.000, 1.8734, 0.6297, 0.9738, 0.2809, 0.0356, -0.1246, -0.5718, 0.9938,
    -1.2219, -0.7730, 1.4148, 1.1016, -0.2054, 0.0367, -3.9128, 0.9156, 6.9750, 0.1774,
    6.4477, -0.1239, -1.5798, -0.5081, -1.7812, 0.1080, 0.2624, 0.0672, -0.2190, -0.4285,
    -1.1000, -0.2515, 0.8952, 0.0156, 0.2782, -0.1812, -4.5000, 1.1766, 24.7219, -13.0812,
    -37.7000, 34.8438, -5.0000, 1.5218, 3.9229, -2.6204, -0.0156, 0.1597, 0.4199, -0.5562,
    -0.5484, -0.6654, -0.2672, 0.7117, 0.7234, -0.6219, -5.6812, 2.6297, 33.3389, -18.3000,
    -62.2500, 52.0781, -3.5000, 0.0016, 1.1477, 0.1062, 0.4659, -0.3296, -0.0876, -0.0329,
    -0.6000, -0.3566, -2.5000, 2.3250, 0.2937, 0.0496, -5.6812, 1.8415, 21.0000, -4.7656,
    -21.5906, 7.2492, -3.5000, -0.1554, 1.4062, 0.3988, 0.0032, 0.0766, -0.0656, -0.1294,
    -1.0156, -0.3670, 1.0078, 1.4051, 0.2875, -0.5328, -3.8500, 3.3750, 14.0000, -0.9999,
    -7.1406, 7.5469, -3.4000, -0.1078, -1.0750, 1.5702, -0.0672, 0.4016, 0.3017, -0.4844,
    -1.0000, 0.0211, 0.5025, -0.5119, -0.3000, 0.1922, 0.7023, -1.6317, 19.0000, -5.0000,
    1.2438, -1.9094, -4.0000, 0.0250, 0.3844, 0.2656, 1.0468, -0.3788, -2.4517, 1.4656,
    -1.0500, 0.0289, 0.4260, 0.3590, -0.3250, 0.1156, 0.7781, 0.0025, 31.0625, -14.5000,
    -46.1148, 55.3750, -7.2312, 0.4050, 13.3500, 0.6234, 1.5000, -0.6426, 1.8564, 0.5636
]
# Sky angles (elevation, azimuth) that define the angular grid over the sky dome
_THETA_GRID = [
    84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
    84, 84, 84, 84, 84, 84, 84, 84, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72,
    72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 48, 48, 48, 48,
    48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 36, 36,
    36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 12, 12, 12, 12, 12, 12, 0
]
_PHI_GRID = [
    0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228,
    240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 0, 12, 24, 36, 48, 60, 72, 84, 96, 108,
    120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324,
    336, 348, 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 
    255, 270, 285, 300, 315, 330, 345, 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
    180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 0, 20, 40, 60, 80, 100, 120,
    140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 0, 30, 60, 90, 120, 150, 180, 210,
    240, 270, 300, 330, 0, 60, 120, 180, 240, 300, 0
]
# -------------------------------------------------------------------


class SkyGen:
    """
    Integrated light-source generator, compatible with Radiance-based simulations.

    Handles assignment of broadband or spectral albedo values.

    Initializes with `daylight_params`, a dataclass containing:
      - n_bands: Number of spectral bands (None for luminance-only)
      - albedo: Broadband ground albedo
      - material_def: Radiance material definition for ground
      - ground_radius: Radius of ground ring (m)
      - n_suns: Number of proxy suns (None for single sun)

    """
    SOLARC = 1361.1  # Solar constant (W/m2) - Gueymard 2018

    def __init__(self, daylight_params):
        self.n_bands = daylight_params.n_bands
        self.n_suns = daylight_params.n_suns
        self.material_def = daylight_params.material_def
        self.ground_radius = daylight_params.ground_radius
        # Broadband aggregated albedo for use by the Perez model
        if not 0.0 <= daylight_params.albedo <= 1.0:
            raise ValueError(f"Albedo must be in [0,1], got {daylight_params.albedo!r}")
        self.albedo = daylight_params.albedo
        logger.info(f"Loaded broadband albedo {self.albedo}")

    def __repr__(self):
        return str(self.__dict__) 
    
    def generate_daylight(
        self,
        dt_index: pd.Timestamp,
        comb_df: pd.DataFrame,
        daylight_errors: dict[int, list[str]]
    ) -> tuple[list[str], dict[int, list[str]]]:
        """
        Build and write the Radiance sky+sun description for one timestep.

        It generates a sky scene description using the Perez All-Weather model: 
        1) Applies Perez sky clearness and brightness parameterization functions
        2) Determines sky-specific Perez coefficients a-e 
        3) Calculates relative luminance pattern over all sky segments
        4) Determines solar radiance
        5) Calculates ground brightness
        6) Generates the Radiance solar description of the sun and sky

        Reads DNI/DHI from `comb_df`, computes Perez sky luminance, crafts the
        sun geometry (single or proxy suns), and writes out a `.rad` file via
        `write_rad_file()`. Records any errors in `daylight_errors`.

        Args:
            dt_index (pd.Timestamp): The timezone-aware timestamp for this hour.
            comb_df (pd.DataFrame): Hourly combined data.
            daylight_errors (dict): Error log keyed by daylight hour.

        Returns:
            sky_file (str):  A single-element list with the filepath of the generated `.rad`.
            daylight_errors (dict): The updated error dictionary.
        """
        n_bands = self.n_bands
        n_suns = self.n_suns
        albedo = self.albedo
        # Retrieve the direct normal and diffuse horizontal irradiance componets
        DNI, DHI = comb_df[['DNI', 'DHI']]
        # Retrieve solar position and air mass
        zenith, azimuth, m = comb_df[['Z', 'A', 'm']] 
        # Retrieve the current daylight hour
        DH = comb_df['DH'] 

        # Limits for sky clearness
        skyclearinf = 1.0
        skyclearsup = 12.01
        # Limits for sky brightness
        skybriginf = 0.01  
        skybrigsup = 0.6  

        if dt_index.tz is None:
            raise ValueError("The datetime index must be timezone-aware.")
        local_time = dt_index
        # Sun disk average opening angle (degrees)
        opening_angle = 0.533
        d_sun_earth = pvlib.solarposition.nrel_earthsun_distance(local_time).iloc[0]  # AU
        # Compute the actual sun disk radius
        sun_radius = (opening_angle / 2) * (1 / d_sun_earth)
        logger.debug(f'Solar disk radius: {round(sun_radius, 4)}')

        # Calculate the Earth's orbital eccentricity correction factor
        #   which adjusts the solar constant based on Earth's varying distance from the sun
        #   throughout the year due to the elliptical shape of its orbit        
        year = local_time.year
        month = local_time.month
        day = local_time.day
        #local_time_hours = local_time.hour + local_time.minute/60 + local_time.second/3600

        def jdate(month: int, day: int, year: int) -> int:
            """Calculate Julian date of the given time instance."""
            days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            # Adjust for leap year
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                days_per_month[1] = 29
            # Calculate day number
            daynum = sum(days_per_month[:month - 1]) + day
            return daynum

        day_number = jdate(month, day, year)
        day_angle = 2 * math.pi * (day_number - 1) / 365
        eccentricity = 1.00011 + 0.034221*math.cos(day_angle) + 0.00128*math.sin(day_angle) + \
                       0.000719*math.cos(2*day_angle) + 0.000077*math.sin(2*day_angle)
        
        # Solar position data processing
        Z = math.radians(zenith)
        E = math.radians(90 - Z)
        # Compute the elevation matrix
        elevation_matrix = rotation_matrix('x', 90 - zenith)
        # The azimuth passed into gensdaylit is defined with reference the + y-axis (North)
        #   and measured positive clockwise, while the matrix rotation is defined along the + z-axis
        #   and measured positive anti-clockwise (right-hand rule)
        A = math.radians(azimuth)
        # To corect for this we pass -azimuth
        azimuth_matrix = rotation_matrix('z', -azimuth)
        # Compute the sun direction unit vector after rotation (with reference due North)
        sun_dir = azimuth_matrix @ elevation_matrix @ np.array([0, 1, 0])
        
        def log_error(error_message: str) -> None:
            """Log an error message and append it to the daylight_errors dict."""
            if DH not in daylight_errors:
                daylight_errors[DH] = []
            daylight_errors[DH].append(error_message)
            logger.error(f"DH {DH}: {error_message}")
        
        def check_irradiances(
            direct: float,
            diffuse: float,
            elevation: int
        ) -> Tuple[float, float]:
            """Clamp negative irradiance values and log sky errors."""
            error_message = ''
            try:
                if direct < 0:
                    direct = 0.0
                if diffuse < 0:
                    diffuse = 0.0
                if direct + diffuse == 0 and math.degrees(elevation) > 0:
                    raise ValueError("Zero irradiance at sun elevation > 0, sky error. ")
                if direct > Apollo.SOLARC:
                    raise ValueError("Direct irradiance exceeds solar constant. ")
            except ValueError as e:
                error_message = str(e)
                log_error(error_message)
            return direct, diffuse
        
        DNI, DHI = check_irradiances(DNI, DHI, E)

        # 1) Perez sky parameterization functions:
        # Calculate sky brightness (epsilon) and clearness (delta) based on direct and diffuse irradiance
        sky_brightness = (m * DHI) / (Apollo.SOLARC * eccentricity)
        sky_clearness = ((DHI + DNI)/DHI + 1.041*(Z**3)) / (1 + 1.041*(Z**3)) if DHI > 0 else skyclearsup - 0.001
        logger.debug(f"Sky brightness: {round(sky_brightness, 2)}, and clearness: {round(sky_clearness, 2)}")

        def check_parameterization(
            clearness: float,
            brightness: float
        ) -> Tuple[float, float]:
            """Enforce Perez model bounds on clearness & brightness."""
            error_message = ''
            try:
                if clearness < skyclearinf:
                    error_message += f"Very low sky clearness {clearness}, set to {skyclearinf}. "
                    clearness = skyclearinf
                elif clearness > skyclearsup:
                    error_message += f"Very high sky clearness {clearness}, set to {skyclearsup - 0.001}. "
                    clearness = skyclearsup - 0.001  # Slightly below upper limit
                if brightness < skybriginf:
                    error_message += f"Very low sky brightness {brightness}, set to {skybriginf}. "
                    brightness = skybriginf
                elif brightness > skybrigsup:
                    error_message += f"Very high sky brightness {brightness}, set to {skybrigsup}. "
                    brightness = skybrigsup
                if error_message:
                    log_error(error_message)
                    
            except Exception as e:
                error_message = f"Unexpecetd error: {str(e)}"
                log_error(error_message)
            return clearness, brightness

        sky_clearness, sky_brightness = check_parameterization(sky_clearness, sky_brightness)
        
        def coeff_lum_perez(
            solar_zenith: float,
            epsilon: float,
            Delta: float
        ) -> List[float]:
            """
            Compute the five Perez model coefficients (a-e). They are functions of sky clearness,
            brightness and solar zenith. Here, they are adjusted on the basis of sky brightness
            and zenith.
            """
            Z_ = solar_zenith
            try:
                if epsilon < skyclearinf or epsilon >= skyclearsup:
                    raise ValueError(f"Epsilon {epsilon} in 'coef_lum_perez' is out of range. ")
                if 1.065 < epsilon < 2.8 and Delta < 0.2:
                    Delta = 0.2
                num_lin = get_numlin(epsilon)
                x = [[_PEREZ_COEFF[20*num_lin + 4*i + j] for j in range(4)] for i in range(5)]
                if num_lin:
                    c_perez = [
                        x[i][0] + x[i][1]*Z_ + Delta * (x[i][2] + x[i][3]*Z_) for i in range(5)
                    ]
                else:
                    c_perez = [
                        x[0][0] + x[0][1]*Z_ + Delta*(x[0][2] + x[0][3]*Z_),
                        x[1][0] + x[1][1]*Z_ + Delta*(x[1][2] + x[1][3]*Z_),
                        math.exp((Delta*(x[2][0] + x[2][1]*Z_))**x[2][2]) - x[2][3],
                        -math.exp(Delta*(x[3][0] + x[3][1]*Z_)) + x[3][2] + Delta*x[3][3],
                        x[4][0] + x[4][1]*Z_ + Delta*(x[4][2] + x[4][3]*Z_)
                    ]
                logger.debug(f"Calculated c_perez coefficients: {c_perez}")
                return c_perez
            
            except ValueError as e:
                log_error(str(e))
                return [0, 0, 0, 0, 0]

        def get_numlin(epsilon: float) -> int:
            """Determines the category number based on sky clearness."""
            if epsilon < 1.065:
                return 0
            elif epsilon < 1.230:
                return 1
            elif epsilon < 1.500:
                return 2
            elif epsilon < 1.950:
                return 3
            elif epsilon < 2.800:
                return 4
            elif epsilon < 4.500:
                return 5
            elif epsilon < 6.200:
                return 6
            return 7

        # 2) Calculate sky specific coefficients a-e as a function of sky condition and solar position
        c_perez = coeff_lum_perez(Z, sky_clearness, sky_brightness)

        def theta_phi_to_zeta_gamma(
            theta: float,
            phi: float,
            solar_zenith: float
        ) -> Tuple[float, float]:
            """Convert theta, phi coordinates to zeta and gamma. Zeta represents the zenith angle
            of the considered sky element, while gamma is the angle between the sky element and 
            the position of the sun."""
            theta_ = math.radians(theta)
            phi_ = math.radians(phi)
            zeta = theta_
            Z_ = solar_zenith
            cos_gamma = math.cos(Z_) * math.cos(theta_) + \
                        math.sin(Z_) * math.sin(theta_) * math.cos(phi_)
            if 1 < cos_gamma < 1.1:
                gamma = 0
            elif cos_gamma >= 1.1:
                raise ValueError("Error in calculation of gamma (angle between point and sun)")
            else:
                gamma = math.acos(cos_gamma)
            return zeta, gamma

        def calc_rel_lum_perez(
            zeta: float,
            gamma: float,
            c_perez: List[float]
        ) -> float:
            """
            Computes the relative luminance (lv) of a particular sky element based on the sky
            specific Perez coefficients (c_perez) and the 'CIE Standard Clear Sky'.
            The lv is defined as the luminance ratio between the considered and an arbitrary sky
            element.

            A new term was introduced by Kittler, Darula, Perez 2001, -exp(d*pi/2), which adjusts
            the width of the circumsolar region. This was termed as the 'CIE Standard General Sky'.
            """
            term1 = c_perez[0] * math.exp(c_perez[1]/math.cos(zeta))
            term2 = c_perez[2] * (math.exp(c_perez[3]*gamma))# - math.exp(c_perez[3]*math.pi/2))
            term3 = c_perez[4] * math.cos(gamma)**2
            lv = (1 + term1) * (1 + term2 + term3)
            return lv

        # 3) Calculate the luminance pattern by looping over 145 sky segments
        # theta and phi angle combinations
        lv_mod = np.zeros(145)
        for j in range(145):
            zeta, gamma = theta_phi_to_zeta_gamma(_THETA_GRID[j], _PHI_GRID[j], Z)
            # Calculate relative luminance of each sky patch
            lv_mod[j] = calc_rel_lum_perez(zeta, gamma, c_perez)
        # Integrate the luminance over the sky and normalize
        buffer = sum(lv_mod[i] * math.cos(math.radians(_THETA_GRID[i])) for i in range(145))
        diffuse_norm = buffer * (2.0 * math.pi / 145.0)
        diffuse_norm = DHI / diffuse_norm
        logger.debug(f"Diffuse normalization factor: {diffuse_norm}")

        # 4) Calculate radiance and solid angle subtended by the Sun
        solid_angle = 2 * math.pi * (1 - math.cos(math.radians(sun_radius)))
        solar_rad = DNI / solid_angle    
        logger.debug(f"Solar radiance: {solar_rad}")

        def normsc(S_INTER: int, elevation: float) -> float:
            """Computes a normalization factor based on elevation and sky conditions via
            polynomial approximation.
            """
            nfc = [
                [2.766521, 0.547665, -0.369832, 0.009237, 0.059229],  # Clear sky
                [3.5556, -2.7152, -1.3081, 1.0660, 0.60227],          # Intermediate sky
            ]
            nf = nfc[S_INTER]
            x = (elevation - math.pi / 4.0) / (math.pi / 4.0)
            # Polynomial evaluation
            nsc = nf[4]
            for i in range(3, -1, -1):
                nsc = nsc * x + nf[i]
            return nsc

        if sky_clearness == 1:
            normfactor = 0.777778
        elif sky_clearness >= 6:
            F2 = .274 * (.91 + 10.0 * math.exp(-3.0*(math.pi/2 - E)) + .45*sun_dir[2]**2)
            normfactor = normsc(0, E) / F2 / math.pi
        elif sky_clearness > 1 and sky_clearness < 6:
            F2 = (2.739 + .9891 * math.sin(.3119 + 2.6*E)) * math.exp(-(math.pi/2 - E) * (.4441 + 1.48*E))
            normfactor = normsc(1, E) / F2 / math.pi
 
        # 5) Ground brightness calculated as the product of:
        #    sky luminosity scaled by DHI, a normalization factor, and albedo
        #    This actually ends up being equal to (GHI + diff_norm_factor) * albedo
        Z_br = calc_rel_lum_perez(0, 0, c_perez)
        Z_br *= diffuse_norm
        ground_brightness = Z_br * normfactor
        if sky_clearness > 1:
            ground_brightness += 6.8e-5 / math.pi * solar_rad * sun_dir[2]
        ground_brightness *= albedo
        logger.debug(f"Ground brightness: {round(ground_brightness, 2)}\n")

        c_perez_text = f"{c_perez[0]:.6f} {c_perez[1]:.6f} {c_perez[2]:.6f} {c_perez[3]:.6f} {c_perez[4]:.6f}"
        
        if n_bands:
            # Filename of sun and sky spectral composition
            sun_spec_filename = f"sun_spectrum_N{n_bands}_{dt_index.strftime('%Y%m%d_%H%M%S')}.dat"
            sky_spec_filename = f"sky_spectrum_N{n_bands}_{dt_index.strftime('%Y%m%d_%H%M%S')}.dat"
        
        # 6) Build sun + sky + ground via helpers

        # - sun function block
        sun_spec = (
            f"\n# Create the specfile primitive containing "
            f"the sun's spectral composition\n"
            f"void specfile sun_spectrum\n"
            f"1 skies/spectrum/{sun_spec_filename}\n0\n0\n"
            if n_bands else ""
        )
        sun_text = self._build_sun_text(
            solar_rad=solar_rad,
            sun_dir=sun_dir,
            sun_radius=sun_radius,
            sky_clearness=sky_clearness,
            n_bands=n_bands,
            n_suns=n_suns,
            zenith=zenith,
            azimuth=azimuth
        )
        # - sky function block
        sky_spec = (
            f"# Create the specfile primitive containing "
            f"the sky's spectral composition\n"
            f"void specfile sky_spectrum\n"
            f"1 skies/spectrum/{sky_spec_filename}\n0\n0"
            if n_bands else ""
        )
        sun_sky_text = self._build_sun_sky_text(
            local_time=local_time,
            E=E,
            A=A,
            sky_clearness=sky_clearness,
            sky_brightness=sky_brightness,
            sun_text=sun_text,
            sun_spectrum_text=sun_spec,
            sky_spectrum_text=sky_spec,
            diffuse_norm=diffuse_norm,
            ground_brightness=ground_brightness,
            c_perez_text=c_perez_text,
            sun_dir=sun_dir,
            n_bands=n_bands
        )
        # - ground function block
        N_acro = f"_N{n_bands}" if n_bands else ""
        ground_text = self._build_ground_text(N_acro)
        full_text = sun_sky_text + "\n" + ground_text
        logger.debug("Full sun + sky + ground description\n{}", full_text)

        Ns_acro = f"_Ns{n_suns}" if n_suns else ""
        sky_path = make_path('skies', f"gensdaylit_DH{DH}{N_acro}{Ns_acro}.rad")
        write_rad_file(sky_path, full_text)

        return [sky_path], daylight_errors
    
    def _build_sun_sky_text(
        self,
        local_time: pd.Timestamp,
        E: float,
        A: float,
        sky_clearness: float,
        sky_brightness: float,
        sun_text: str,
        sun_spectrum_text: str,
        sky_spectrum_text: str,
        diffuse_norm: float,
        ground_brightness: float,
        c_perez_text: str,
        sun_dir: np.ndarray,
        n_bands: int
    ) -> str:
        """Generate the multi-line Perez sky + sun description."""
        tpl = textwrap.dedent("""\
            # Local datetime: {local_time}

            # Solar elevation, azimuth: {elev:.1f} {azi:.1f}
            # ε (clearness), Δ (brightness): {clearness:.3f} {brightness:.3f}
            {sun_spec}
            {sun_text}
            {sky_spec}

            # Perez sky brightness function
            {modifier} brightfunc skyfunc
            2 skybright perezlum.cal
            0
            10 {diffuse:.6e} {ground:.6e} {cperez}
            {sdx:.6f} {sdy:.6f} {sdz:.6f}

            # Apply the{spectral} skyfunc modifier to the grey glow
            skyfunc glow sky_glow
            0
            0
            4 1 1 1 0

            # Define the sky hemisphere
            sky_glow source sky
            0
            0
            4 0 0 1 180
        """)
        return tpl.format(
            local_time=local_time,
            elev=math.degrees(E),
            azi=math.degrees(A),
            clearness=sky_clearness,
            brightness=sky_brightness,
            sun_spec=sun_spectrum_text,
            sun_text=sun_text,
            sky_spec=sky_spectrum_text,
            modifier="sky_spectrum" if n_bands else "void",
            diffuse=diffuse_norm,
            ground=ground_brightness,
            cperez=c_perez_text,
            sdx=sun_dir[0], sdy=sun_dir[1], sdz=sun_dir[2],
            spectral=" (spectral)" if n_bands else ""
        )
        
    def _build_sun_text(
        self,
        solar_rad: float,
        sun_dir: np.ndarray,
        sun_radius: float,
        sky_clearness: float,
        n_bands: int,
        n_suns: Optional[int],
        zenith: float,
        azimuth: float
    ) -> str:
        """Generate Radiance text for single or multiple proxy suns.

        Args:
            solar_rad: Radiance of the sun (W/m²·sr).
            sun_dir:  Unit direction vector of the sun center.
            sun_radius: Angular radius of the solar disk (deg).
            sky_clearness: Perez sky clearness parameter.
            n_bands: Number of spectral bands (None for luminance-only).
            n_suns: Number of proxy suns (None for single sun).
            zenith: Solar zenith angle (deg).
            azimuth: Solar azimuth angle (deg).

        Returns:
            Multi-line Radiance description of sun source(s).
        """
        diameter = sun_radius * 2
        material = "sun_spectrum" if n_bands else "void"
        # Radiance (software) operates in RGB (3 channels)
        radiance_val = (
            f"3 {solar_rad:.6e} {solar_rad:.6e} {solar_rad:.6e}"
            if sky_clearness > 1 else
            "3 0.0 0.0 0.0"
        )

        if n_suns is None:
            tpl = textwrap.dedent("""\
                {material} light radiance
                0
                0
                {radiance}

                radiance source sun
                0
                0
                4 {sx:.6f} {sy:.6f} {sz:.6f} {diameter:.6f}
            """)
            return tpl.format(
                material=material,
                radiance=radiance_val,
                sx=sun_dir[0], sy=sun_dir[1], sz=sun_dir[2],
                diameter=diameter
            )

        # proxy-sun distribution
        distributed = solar_rad / n_suns
        mag_val = (
            f"3 {distributed:.6e} {distributed:.6e} {distributed:.6e}"
            if sky_clearness > 1 else
            "3 0.0 0.0 0.0"
        )
        mag_tpl = textwrap.dedent("""\
            {material} light radiance
            0
            0
            {mag_val}
        """)
        text = mag_tpl.format(material=material, mag_val=mag_val)

        cache = f"sun_displacement_vectors_Ns{n_suns}.rad"
        dirs = self._distribute_proxy_suns(sun_radius, zenith, azimuth, cache)

        # Iterate through proxy suns and append them to the Radiance sun description
        for i, d in enumerate(dirs):
            text += textwrap.dedent(f"""\

                radiance source sun{i}
                0
                0
                4 {d[0]:.6f} {d[1]:.6f} {d[2]:.6f} {diameter:.6f}
            """)
        return text

    def _build_ground_text(self, N_acro: str) -> str:
        """Generate the Radiance ground-plane ring text."""
        tpl = textwrap.dedent("""\
            # Use an upside-down sky to represent ground
            skyfunc glow ground_glow
            0
            0
            4 1 1 1 0

            ground_glow source ground
            0
            0
            4 0 0 -1 180

            # Ground plane locally replaces ground glow with a material whose
            # irradiance depends on the surrounding environment and sky
            {material}{N_acro} ring ground_plane
            0
            0
            8 0 0 -.01 0 0 1 0 {radius}
        """)
        return tpl.format(
            material=self.material_def,
            N_acro=N_acro,
            radius=self.ground_radius
        )

    def _distribute_proxy_suns(
        self,
        sun_disk_radius: float,
        sun_zenith: float,
        sun_azimuth: float,
        cache_file: str
    ) -> list[np.ndarray]:
        """
        Build or load proxy-sun direction vectors over the solar disk.

        Either loads a cached Nx2 displacement file, or computes an equal-area
        Fibonacci lattice, then transforms into unit direction vectors.
      
        Arguments:
            sun_disk_radius (float): Solar disk angular radius (deg) in the sky dome.
            sun_zenith (float): Solar zenith angle (degrees) used in point source calculations.
            sun_azimuth (float): Solar azimuth angle (degrees) used in point source calculations.
            cache_file (str): Filename under 'skies' for storing displacement vectors.
            
        Returns:
            sun_dir_list (list): Contains the sun direction vectors for each proxy sun.
        """
        filepath = make_path('skies', cache_file)
        # Check if the file with the coordinates of the proxy suns exists
        if os.path.exists(filepath):
            points = np.loadtxt(filepath)
            if points.ndim == 1:
                points = points[np.newaxis, :]
            points_cartesian = points
        else:
            # Calculate the displacement vectors
            golden_ratio = (1 + np.sqrt(5)) / 2
            epsilon = 0.5  # For Canonical Lattice
            # Each point, i, is defined based on an arbitrary number, n_suns
            i = np.arange(0, self.n_suns)
            # Definition of cartesian coordinates
            x = i / golden_ratio
            y = (i + epsilon) / (self.n_suns - 1 + 2*epsilon)  # Fibonacci Lattice
            # Map the point distribution to a unit disk via equal-area transformation
            theta = 2 * np.pi * x  # [0, 2pi]
            r = np.sqrt(y) * np.radians(sun_disk_radius)  # Adjust by solar disk radius
            # Convert back to cartesian
            x_coord = r * np.cos(theta)
            y_coord = r * np.sin(theta)
            # Stack arrays vertically, consisting of each point's coordinates (n rows, 2 columns)
            points_cartesian = np.vstack((x_coord, y_coord)).T
            # Save the displacement vectors to the cache file
            np.savetxt(filepath, points_cartesian, fmt="%.6f")
        
        # Compute the proxy sun direction vectors given the actual solar azimuth and zenith angles
        sun_dir_list = []
        for dx, dy in points_cartesian:
            elevation_angle = 90 - sun_zenith - np.degrees(dy)
            azimuth_angle = -sun_azimuth + np.degrees(dx)
            elevation_matrix = rotation_matrix('x', elevation_angle) 
            azimuth_matrix = rotation_matrix('z', azimuth_angle)
            sun_dir = azimuth_matrix @ elevation_matrix @ np.array([0, 1, 0])
            sun_dir_list.append(sun_dir)

        return sun_dir_list     
