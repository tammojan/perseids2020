#!/usr/bin/env python
""" Make plots of AARTFAAC fits files with an overlaid CAMS trajectory """

import pandas as pd
import numpy as np

from glob import glob
import astropy.units as u

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

import astropy
from astropy.io import fits
from astropy.wcs import WCS

import astropy.units as u
from astropy.time import Time
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import EarthLocation, AltAz, GCRS, ITRS, Angle, SkyCoord, FK5

from tqdm import tqdm

import warnings

nsteps = 21  # points along cams meteor trajectory to plot

camsdata_full = pd.read_fwf("SummaryMeteorLog CAMS Benelux 120820 .txt",
                            index_col=0,
                            header=[0, 1],
                            skip_blank_lines=True,
                            skiprows=[2],
                            skipinitialspace=True)

camsdata_full.columns = [
    ' '.join(col) if isinstance(col, tuple) else col
    for col in camsdata_full.columns
]

camsdata_full["astropytime_beg"] = Time(
    list(camsdata_full["Observed Date"] + " " + camsdata_full["Ref Time UT"])
) + np.array(camsdata_full["Tbeg sec"]) * u.s
camsdata_full["astropytime_end"] = Time(
    list(camsdata_full["Observed Date"] + " " + camsdata_full["Ref Time UT"])
) + np.array(camsdata_full["Tend sec"]) * u.s

loc_lofar = EarthLocation(lat=52.9153 * u.deg,
                          lon=6.8698 * u.deg,
                          height=20 * u.m)


def llh_to_radec(lon, lat, height, obstime, location):
    """Convert latitude, longitude, height to apparent Ra/Dec at a given site"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        altaz_near = AltAz(
            az=0 * u.deg,
            alt=90 * u.deg,
            distance=height,
            obstime=obstime,
            location=EarthLocation(lon=lon, lat=lat, height=0),
        ).transform_to(AltAz(location=location, obstime=obstime))

        altaz_far = AltAz(location=location,
                          obstime=obstime,
                          alt=altaz_near.alt,
                          az=altaz_near.az)

        return altaz_far.transform_to(FK5)


camsdata = camsdata_full  #[camsdata_full['Int-mV mag'] <= 0]

fitsnames = sorted(glob("A12_meteor/*.fits"))

fitstimes = Time(
    [Time(fitsname.split("_")[-2][-21:]) for fitsname in fitsnames])

camstimes = Time(camsdata['astropytime_beg'])

# rows in CAMS that have an AARTFAAC fits file within 3 seconds
matching_rownrs = camsdata.iloc[np.where(
    np.min(np.abs((fitstimes[:, np.newaxis] -
                   camstimes[np.newaxis, :]).to(u.s)),
           axis=0) < 2 * u.s)[0]].index

for camsindex in tqdm(matching_rownrs[146:]):
    row = camsdata.loc[camsindex]

    lon = np.linspace(row["LonBeg +E deg"] * u.deg,
                      row["LonEnd +E deg"] * u.deg, nsteps)
    lat = np.linspace(row["LatBeg +N deg"] * u.deg,
                      row["LatEnd +N deg"] * u.deg, nsteps)
    height = np.linspace(row["Hbeg km"] * u.km, row["Hend km"] * u.km, nsteps)
    obstime = Time(
        np.linspace(row["astropytime_beg"].mjd, row["astropytime_end"].mjd,
                    nsteps),
        format="mjd",
    )
    radec = llh_to_radec(lon, lat, height, obstime, location=loc_lofar)

    fitsname = fitsnames[np.argmin(np.abs(row['astropytime_beg'] - fitstimes))
                         + 1]  # 2 seconds after the CAMS meteor

    hdu = fits.open(fitsname)[0]

    wcs = WCS(hdu.header)

    (x, y) = skycoord_to_pixel(radec, wcs, 0)

    fig, ax0 = plt.subplots(1, 1, figsize=(10, 10))
    fig.add_axes(ax0)
    data = hdu.data[0, 0].copy()
    roi = data[int(min(y)):int(max(y)), int(min(x)):int(max(x))]

    if roi.size == 0:
        print("Empty roi for", camsindex)
        plt.close()
        continue
    ax0.imshow(data,
               origin='lower',
               vmin=np.percentile(data, 1),
               vmax=np.percentile(data, 99.9))
    ax0.set_axis_off()
    ax0.set_aspect(1, adjustable='datalim')

    ax2 = plt.axes([0, 0, 1, 1])
    ip2 = InsetPosition(ax0, [1.02, 0.0, 0.3, 0.3])
    ax2.set_axes_locator(ip2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(data,
               origin='lower',
               vmin=np.percentile(roi, 1),
               vmax=np.percentile(roi, 99))
    ax2.set_xlim(min(x) - 100, max(x) + 100)
    ax2.set_ylim(min(y) - 100, max(y) + 100)
    ax2.set_aspect(1, adjustable='datalim')

    ax1 = plt.axes([0, 0, 1, 1], label="twee")
    ip1 = InsetPosition(ax0, [1.02, 0.32, 0.3, 0.3])
    ax1.set_axes_locator(ip1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(data,
               origin='lower',
               vmin=np.percentile(roi, 1),
               vmax=np.percentile(roi, 99))
    ax1.set_xlim(min(x) - 100, max(x) + 100)
    ax1.set_ylim(min(y) - 100, max(y) + 100)
    ax1.set_aspect(1, adjustable='datalim')

    ax2.plot(x, y, color='red', linewidth=0.7)
    ax2.plot(x[:1], y[:1], 'rx')

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()

    rect = patches.Rectangle((xmin, ymin),
                             xmax - xmin,
                             ymax - ymin,
                             linewidth=0.5,
                             edgecolor='k',
                             facecolor='none')
    ax0.add_patch(rect)

    label = '#{} (Mag {:.1f}): {} / {}'.format(
        camsindex, camsdata.loc[camsindex]["Int-mV mag"],
        camsdata.loc[camsindex]["astropytime_beg"].isot[:21], fitsname[11:])
    ax0.text(0.02, 0.98, label, color='white', transform=ax0.transAxes)
    fig.savefig("{}.png".format(camsindex), bbox_inches='tight')
    plt.close()
