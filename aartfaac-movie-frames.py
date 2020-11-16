#!/usr/bin/env python
# coding: utf-8

import sys

import pandas as pd
import numpy as np

from shapely.geometry import LineString

from glob import glob
import astropy.units as u

import matplotlib.patches as patches
import matplotlib.pyplot as plt

import urllib.request

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

camsdata = pd.read_fwf("../SummaryMeteorLog CAMS Benelux 120820 .txt",
                            index_col=0,
                            header=[0, 1],
                            skip_blank_lines=True,
                            skiprows=[2],
                            skipinitialspace=True)

camsdata.columns = [
    ' '.join(col) if isinstance(col, tuple) else col
    for col in camsdata.columns
]

camsdata["astropytime_beg"] = Time(
    list(camsdata["Observed Date"] + " " + camsdata["Ref Time UT"])
) + np.array(camsdata["Tbeg sec"]) * u.s
camsdata["astropytime_end"] = Time(
    list(camsdata["Observed Date"] + " " + camsdata["Ref Time UT"])
) + np.array(camsdata["Tend sec"]) * u.s

camsdata = camsdata.join(pd.read_csv("aartfaac-cams-matches.csv", index_col=0))

loc_lofar = EarthLocation(lat=52.9153 * u.deg,
                          lon=6.8698 * u.deg,
                          height=20 * u.m)


def llh_to_radec(lon, lat, height, obstime, location):
    """Convert latitude, longitude, height to apparent Ra/Dec at a given site"""

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

def makerect(x, y):
    xmin = max(0, np.min(x) - 100)
    ymin = max(0, np.min(y) - 100)
    xmax = min(np.max(x) + 100, hdu.data.shape[-1])
    ymax = min(np.max(y) + 100, hdu.data.shape[-1])
    return xmin, ymax, patches.Rectangle((xmin, ymin),
                                         xmax - xmin,
                                         ymax - ymin,
                                         linewidth=0.5,
                                         edgecolor='red',
                                         facecolor='none')


fitsnames = sorted(glob("/data1/dijkema/A12_meteor/*.fits"))

camstimes = Time(camsdata['astropytime_beg'])

previous_fitsname = ''

start_index = 0
end_index = -1

if len(sys.argv) > 1:
    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])

for frame_nr, fitsname in tqdm(enumerate(fitsnames[start_index: end_index]), total=len(fitsnames[start_index: end_index])):

    # Sky _16B if _15B is present (etc), to get only one frame per timeslot
    if fitsname[:47] == previous_fitsname[:47]:
        continue
    else:
        previous_fitsname = fitsname

    hdu = fits.open(fitsname)[0]

    fitstime = Time(hdu.header["DATE-OBS"])

    camsmatches = camsdata.iloc[((camstimes - fitstime).to(u.s) < 2 * u.s)
                                & ((camstimes - fitstime).to(u.s) > -60 * u.s)]

    wcs = WCS(hdu.header)

    fig = plt.figure(figsize=(10.8, 10.8)) # Will yield 1080x1080 frames
    ax = plt.axes([0, 0, 1, 1])
    fig.add_axes(ax)
    ax.set_axis_off()
    data = hdu.data[0, 0]
    ax.set_aspect(1, adjustable='datalim')
    ax.imshow(data,
              origin='lower',
              vmin=np.percentile(data, 1),
              vmax=np.percentile(data, 99.9))

    ax.text(0.02,
            0.94,
            fitsname.split("/")[-1][:-5],
            color='white',
            transform=ax.transAxes)

    for rownr, row in camsmatches.iterrows():
        try:
            lon = np.linspace(row["LonBeg +E deg"] * u.deg,
                              row["LonEnd +E deg"] * u.deg, nsteps)
            lat = np.linspace(row["LatBeg +N deg"] * u.deg,
                              row["LatEnd +N deg"] * u.deg, nsteps)
            height = np.linspace(row["Hbeg km"] * u.km, row["Hend km"] * u.km,
                                 nsteps)
            obstime = Time(
                np.linspace(row["astropytime_beg"].mjd,
                            row["astropytime_end"].mjd, nsteps),
                format="mjd",
            )
            radec = llh_to_radec(lon, lat, height, obstime, location=loc_lofar)
            (x, y) = skycoord_to_pixel(radec, wcs, 0)
            xmin, ymax, rect = makerect(x, y)
            if row["AARTFAAC"] is not None and row["AARTFAAC"] > 0:
                color = 'white'
            else:
                color = 'red'
            alpha = np.interp(
                (fitstime - row["astropytime_beg"]).to(u.s).value, (0, 60),
                (1, 0))

            plt.plot(*(LineString(np.array([x, y]).T).buffer(20).exterior.xy),
                     color="red", linewidth=0.5, alpha=alpha)

            rect.set_alpha(alpha)
            rect.set_edgecolor(color)
            ax.add_patch(rect)
            ax.text(xmin,
                    ymax - 31,
                    f"#{rownr}: {row['AARTFAAC']}",
                    color='white',
                    alpha=alpha)
            ax.set_xlim(0, data.shape[1])
            ax.set_ylim(0, data.shape[0])
        except:
            print("Problem with #" + str(rownr) + ", " + fitsname)
            raise
            continue
    fig.savefig(f"/data1/dijkema/png/{frame_nr + start_index:05d}.png", dpi=100)
    plt.close()

# ffmpeg -framerate 10 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
