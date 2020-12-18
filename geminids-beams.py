#!/usr/bin/env python
# coding: utf-8

import astropy.units as u
from astropy.time import Time
import numpy as np

from tqdm.autonotebook import tqdm

from scipy.stats import linregress

import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


from astropy.io import fits

subbands = [156, 165, 174, 187, 195, 213, 221, 231, 243, 256, 257, 267, 278, 284, 296, 320]
# Put SB256, SB257 at the bottom
reorder = [9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]


rs305_files = [f"L{i}" for i in range(801044, 801101, 4)]


de605_files = [f"L{i}" for i in range(801104, 801161, 4)]


rs305_files = ["L801044", "L801048", ]
de605_files = ["L801108"]

for hour in tqdm(range(len(rs305_files))):
    f = {}
    g = {}
    for sap in range(3):
        f[sap] = fits.open(f"/data1/geminids/{rs305_files[hour]}_SAP00{sap}_B000.fits")[0]
        g[sap] = fits.open(f"/data1/geminids/{de605_files[hour]}_SAP00{sap}_B000.fits")[0]
    
    t0 = Time(f[0].header['MJD-OBS'], format='mjd')
    alltimes = t0 + f[0].header['CDELT1'] * np.arange(f[0].data.shape[-1]) * u.s
    
    for starttime in range(5,6):
        start = max(round(((starttime*10-1)*u.min / (f[0].header["CDELT1"] * u.s)).si.value), 0)
        num = round((12*u.min / (f[0].header["CDELT1"] * u.s)).si.value)
        fig = plt.figure(figsize=(35, 35))
        gridspec = GridSpec(18, 3)
        gridspec.update(wspace=0.03, hspace=0)
        times = alltimes[start:start+num]

        ylims = {}
        for sap in range(3):
            # RS305
            for i in range(15, -1, -1):
                ax = plt.subplot(gridspec[15-i, sap])
                data = f[sap].data[0, reorder[i]*64: (reorder[i]+1)*64].mean(axis=0)[start: start+num]

                mask = abs(data - np.mean(data[data!=0.])) < 3 * np.std(data[data!=0.])
                slope, intercept, _, _, _ = linregress(times.mjd[mask], data[mask])
                fitted = intercept + slope * times.mjd

                ax.plot(times.datetime, data-fitted, color='navy', markersize=2, lw=0.15)
                ax.set_yticks([])
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.grid(lw=0.5)
                if not ax.is_last_row():
                    ax.set_xticklabels([''] * len(ax.get_xticklabels()))
                if ax.is_first_col():
                    ax.text(0.02, 0.5, f"SB{subbands[reorder[i]]}", rotation=90, transform=ax.transAxes, verticalalignment='center')
                    ax.set_ylim((np.quantile((data-fitted)[mask], 0.002), np.quantile((data-fitted)[mask], 0.999)))
                    ylims[i] = ax.get_ylim()
                else:
                    ax.set_ylim(ylims[i])

            # DE605 SB256
            if sap == 1:
                ax = plt.subplot(gridspec[16, sap])
                data = g[sap].data[0, 64:128].mean(axis=0)[start: start+num][:len(times)]

                mask = abs(data - np.mean(data[data!=0.])) < 3 * np.std(data[data!=0.])
                slope, intercept, _, _, _ = linregress(times.mjd[mask[:len(times)]], data[mask])
                fitted = intercept + slope * times.mjd

                ax.plot(times.datetime, data-fitted, color='navy', markersize=2, lw=0.15)
                ax.set_ylim((np.quantile((data-fitted)[mask], 0.002), np.quantile((data-fitted)[mask], 0.9999)))
                ax.set_yticks([])
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.grid(lw=0.5)
                ax.text(0.02, 0.5, "DE605 SB256", rotation=90, transform=ax.transAxes, verticalalignment='center')

            # DE605 subband 174
            ax = plt.subplot(gridspec[17, sap])
            data = g[sap].data[0, :64].mean(axis=0)[start: start+num][:len(times)]

            mask = abs(data - np.mean(data[data!=0.])) < 3 * np.std(data[data!=0.])
            slope, intercept, _, _, _ = linregress(times.mjd[mask[:len(times)]], data[mask])
            fitted = intercept + slope * times.mjd

            ax.plot(times.datetime, data-fitted, color='navy', markersize=2, lw=0.15)
            ax.set_ylim((np.quantile((data-fitted)[mask], 0.002), np.quantile((data-fitted)[mask], 0.999)))
            ax.set_yticks([])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.grid(lw=0.5)
            if ax.is_first_col():
                ax.text(0.02, 0.5, "DE605 SB174", rotation=90, transform=ax.transAxes, verticalalignment='center')
                ax.set_ylim((np.quantile((data-fitted)[mask], 0.002), np.quantile((data-fitted)[mask], 0.999)))
                ylims[17] = ax.get_ylim()
            else:
                ax.set_ylim(ylims[17])
        fig.savefig(f"{hour}_{starttime}.png", bbox_inches='tight')
        plt.close('all')
