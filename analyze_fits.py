from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import scipy.signal

import seaborn as sns

from pandas.plotting import register_matplotlib_converters

from astropy.time import Time
import astropy.units as u
from astropy import wcs

from tqdm.autonotebook import tqdm

import warnings


# In[2]:


import sys
sys.path.append("/home/dijkema/opt/lofar_dynspec_tools/")


# In[3]:


from channelize import channelize


# In[4]:


from ipywidgets import interact, interactive


# In[5]:


import pandas as pd


# In[6]:


df = pd.read_csv("https://github.com/tammojan/perseids2020/raw/master/meteor-overview.csv")
df.set_index(pd.DatetimeIndex(df['Time']), inplace=True)


# In[7]:


from astropy.timeseries import TimeSeries


# In[8]:


times_allsky = TimeSeries.from_pandas(df[df['DAARO! All Sky Camera'] > 0]).time


# In[9]:


times_lofar = TimeSeries.from_pandas(df[df['Watec optical Burlage+Dwingeloo Pointing LOFAR'] > 0]).time


# In[10]:


times_twist = TimeSeries.from_pandas(df[df['Watec optical Burlage+Dwingeloo Pointing Twist'] > 0]).time


# In[11]:


times_aartfaac = TimeSeries.from_pandas(df[pd.to_numeric(df['AARTFAAC'], errors='coerce') > 0]).time


# In[12]:


camsdata = np.loadtxt("SummaryMeteorLog CAMS Benelux 120820 .txt",
               skiprows=3,
               usecols=(0, 1, 2, 3, 4, 5, 7, 9, 15, 17, 19, 21, 23, 25, 31),
               dtype={"names": ("number", "date", "time", "tbeg", "tend", "RAinf", "DECinf", "Vinf",
                                "LatBeg", "LonBeg", "Hbeg", "LatEnd", "LonEnd", "Hend", "MaxmV"),
                      "formats": ("f4", "S10", "S11", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f2")})


# In[13]:


with open("SummaryMeteorLog CAMS Benelux 120820 .txt") as f:
    colnames = f.readline().split()[2:]
for colnum, colname in enumerate(colnames):
    if colname == "+/-":
        colnames[colnum] = colnames[colnum-1] + " " + colname
cams_data = pd.read_csv("SummaryMeteorLog CAMS Benelux 120820 .txt", delim_whitespace=True, names=colnames, skiprows=[0,1,2])
cams_data["Time"] = cams_data["Ref"] + " " + cams_data["Time"]
cams_data["astropytime"] = Time(list(cams_data["Time"].values))


# In[14]:


times_cams = Time(cams_data["astropytime"].values)


# In[17]:


sns.set_style("white")
sns.set_context("notebook", font_scale=1)
register_matplotlib_converters()


# In[18]:


sap0 = fits.open("/home/dijkema/L792714_SAP000_B000_P000.fits")[0]
sap1 = fits.open("/home/dijkema/L792714_SAP001_B000_P000.fits")[0]
#sap0_hires = fits.open("/home/dijkema/L792714_SAP000_B000_P000.fits")[0]


# In[19]:


sap0.data.shape


# In[20]:


def channel_to_freq(channel, header):
    """Return centre frequency for a given channel number"""
    return ((header['CRVAL2'] + channel * header['CDELT2']) * u.Hz).to(u.MHz)


# In[21]:


def freq_to_channel(freq, header):
    """Return centre frequency for a given channel number"""
    return round(((freq - header['CRVAL2'] * u.Hz) / (header['CDELT2'] * u.Hz)).si.value)


# In[67]:


def show(data, header, chan_nr, i=0, step=60*u.s, central_time=None, ax=None):
    t0 = Time(header['MJD-OBS'], format='mjd') + header['CRVAL1'] * u.s
    delta_t = header['CDELT1'] * u.s
    ntimes = round((step/delta_t).si.value)
    if central_time is not None:
        begin_time_idx = max(int((central_time - t0).to(u.s) / delta_t - ntimes/2), 0)
    else:
        begin_time_idx = i * ntimes
    end_time_idx = begin_time_idx + ntimes
    mydata = data
    if len(mydata.shape) == 3:
        mydata = mydata[0]
    mydata = mydata[chan_nr, begin_time_idx:end_time_idx]
    times = t0 + (np.arange(0, mydata.shape[-1]) + begin_time_idx) * delta_t
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(24,3))
    ax.plot(times.datetime, mydata)
    peaks, peak_heights = scipy.signal.find_peaks(mydata, prominence=3e6, distance=40)
    ax.scatter([times.datetime[idx] for idx in peaks], 0*peaks, color='red')
    ax.set_aspect('auto')
    ax.set_xlim(np.min(times).datetime, np.max(times).datetime)
    ax.set_yticks([])
    #ax.set_ylim(0, np.percentile(mydata, 99.5))
    ax.set_title(f"Channel {chan_nr}")
    if central_time is not None:
        ax.vlines(central_time.datetime, 0, np.max(mydata))
    return [times[i].isot for i in peaks]


# In[23]:


def find_channel_peaks(data):
    """Compute the number of peaks per channel"""
    num_peaks = []
    for chan_nr in tqdm(range(data.shape[0])):
        peaks, peak_heights = scipy.signal.find_peaks(data[chan_nr, :], prominence=3e6, distance=40)
        num_peaks.append(len(peaks))
    return num_peaks


# In[24]:


dourbes_peaks_sap0, peak_heights = scipy.signal.find_peaks(sap0.data[172, :], prominence=3e6, distance=40)
ieper_peaks_sap0, peak_heights = scipy.signal.find_peaks(sap0.data[185, :], prominence=3e6, distance=40)
dourbes_peaks_sap1, peak_heights = scipy.signal.find_peaks(sap1.data[172, :], prominence=3e6, distance=40)
ieper_peaks_sap1, peak_heights = scipy.signal.find_peaks(sap1.data[185, :], prominence=3e6, distance=40)


# In[25]:


all_peaks = np.concatenate([dourbes_peaks_sap0, ieper_peaks_sap0, dourbes_peaks_sap1, ieper_peaks_sap1])


# In[26]:


chan_peaks_sap0 = find_channel_peaks(sap0.data)
chan_peaks_sap1 = find_channel_peaks(sap1.data)


# In[27]:


fig, ax0 = plt.subplots(1, 1, figsize=(12, 6))
peaks0, _ = scipy.signal.find_peaks(chan_peaks_sap0, prominence=40)
annotations = {172: "\n(Dourbes)", 185: "\n(Ieper)"}
ax0.plot(channel_to_freq(np.arange(256), sap0.header).value, chan_peaks_sap0);
for peak_idx in peaks0:
    ax0.annotate(f"{channel_to_freq(peak_idx, sap0.header):.3f}" + annotations.get(peak_idx, ''),
                 (channel_to_freq(peak_idx, sap0.header).value, chan_peaks_sap0[peak_idx]),
                 fontsize=10)
ax0.vlines(channel_to_freq(128, sap0.header).value, 0, 700, linewidth=1)
ax0.set_title("Number of peaks per channel (SB255 + SB256)", fontsize=16);
ax0.annotate("SB255", (channel_to_freq(110, sap0.header).value, 600))
ax0.annotate("SB256", (channel_to_freq(132, sap0.header).value, 600))
ax0.set_xlabel("Channel frequency")
ax0.set_ylabel("Number of peaks")
fig.tight_layout()


# In[95]:


def show2d(data, header, chan_nr=None, i=0, step=60*u.s, central_time=None, ax=None, num_channels=None, maxquantile=.9999, drawline=True):
    if chan_nr is None and num_channels is None:
        begin_chan_idx = 0
        end_chan_idx = -1
    else:
        begin_chan_idx = max(chan_nr - num_channels//2, 0)
        end_chan_idx = min(chan_nr + num_channels//2, data.shape[-2] - 1)

    t0 = Time(header['MJD-OBS'], format='mjd') + header['CRVAL1'] * u.s
    delta_t = header['CDELT1'] * u.s
    
    ntimes = round((step/delta_t).si.value)

    if central_time is not None:
        begin_time_idx = max(int((central_time - t0).to(u.s) / delta_t - ntimes/2), 0)
    else:
        begin_time_idx = i * ntimes
    end_time_idx = begin_time_idx + ntimes
    mydata = data
    if len(mydata.shape) == 3:
        # Strip polarization
        mydata = mydata[0]

    mydata = mydata[begin_chan_idx:end_chan_idx, begin_time_idx:end_time_idx]

    times = t0 + (np.arange(0, mydata.shape[-1]) + begin_time_idx) * delta_t

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(12,6))
    else:
        fig = None
    ax.imshow(mydata, extent=[mdates.date2num(times[0].datetime),
                              mdates.date2num(times[-1].datetime),
                              channel_to_freq(begin_chan_idx - 0.5, header).value,
                              channel_to_freq(end_chan_idx - 0.5, header).value],
              interpolation='nearest',
              vmin=np.quantile(mydata, 0),
              vmax=np.quantile(mydata, maxquantile),
              origin='lower')
    ax.xaxis_date()
    ax.set_ylabel("Frequency (MHz)")
    #peaks, peak_heights = scipy.signal.find_peaks(mydata, prominence=1e6, distance=10)
    #ax.scatter([times.datetime[idx] for idx in peaks], 0*peaks, color='red')
    ax.set_aspect('auto')
    ax.set_title(f"Channel {chan_nr}")
    ax.ticklabel_format(axis='y', useOffset=False)
    if central_time is not None and drawline:
        ax.vlines(central_time.datetime,
                  channel_to_freq(begin_chan_idx - 0.5, header).value,
                  channel_to_freq(end_chan_idx - 0.5, header).value, 'white', linewidth=0.5)
        #ax.annotate(central_time.isot[:-4], (central_time.datetime, channel_to_freq(begin_chan_idx - 0.5, header).value), color='white', rotation=90, fontsize=10)
    return fig, ax


# In[29]:


fig, ax = show2d(sap0.data, sap0.header, 192, num_channels=128, central_time=Time("2020-08-12 22:14:09.140"))
ax.set_title("Typical Dourbes and Ieper reflections at 22:14:09 matching CAMS detection #147");
ax.annotate("Dourbes (49.970MHz)", (Time("2020-08-12T22:14:20").datetime, 49.970), color='white', fontsize=10, xytext=(10, -12), textcoords='offset pixels')
ax.annotate("Ieper (49.990MHz)", (Time("2020-08-12T22:14:20").datetime, 49.990), color='white', fontsize=10, xytext=(10, -12), textcoords='offset pixels');
#fig.savefig("reflection147.png")


# In[32]:


def myshow(i):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 10))
    _ = show2d(sap0.data, sap0.header, 192, ax=ax0, num_channels=128, central_time=times_cams[i])
    _ = show2d(sap1.data, sap1.header, 192, ax=ax1, num_channels=128, central_time=times_cams[i])
    ax0.set_title(f"SAP0 (pointing to LOFAR)")
    ax1.set_title(f"SAP1 (pointing to Twist)")
    for ax in ax0, ax1:
        vmin, vmax = ax.get_images()[0].get_clim()
        ax.get_images()[0].set_clim(vmin + (vmax - vmin) * 0, vmax - (vmax - vmin)*0)


# In[33]:


def myshow1d(i):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(20, 10))
    _ = show(sap0.data, sap0.header, 172, i, ax=ax0)
    _ = show(sap1.data, sap0.header, 172, i, ax=ax1)
    ax0.set_title(f"SAP0 (pointing to LOFAR)")
    ax1.set_title(f"SAP1 (pointing to Twist)")


# In[34]:


interact(myshow, i=(0, len(times_cams)));


# In[35]:


#obs = fits.open("/home/dijkema/dwingeloo_lofar/2020-08-13T02:22:55.582.fits")[0]
obs = fits.open("/home/dijkema/burlage_lofar/2020-08-13T02:22:52.655.fits")[0]
#obs = fits.open("/home/dijkema/dwingeloo_lofar/2020-08-13T00:55:25.459.fits")[0]
#obs = fits.open("/home/dijkema/burlage_lofar/2020-08-13T00:55:22.724.fits")[0]

fig = plt.figure(figsize=(15, 15))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mywcs = wcs.WCS(obs.header).celestial

ax = fig.add_subplot(2, 2, 1, projection=mywcs)
img = ax.imshow(obs.data[2]);
ax.grid(linewidth=.1)
ax.tick_params(direction='in')
ax.set_title(f"Observed from , {obs.header['DATE-OBS'][:-4]}");


# In[36]:


all_peak_times = Time(sap0.header['MJD-OBS'], format='mjd') + sap0.header['CRVAL1'] * u.s + sap0.header['CDELT1'] * u.s * all_peaks


# In[37]:


cams_data["hasmatch"] = np.min(np.abs((all_peak_times - Time(list(cams_data["Time"].values))[:, np.newaxis]).to(u.s).value), axis=1) < 3


# In[39]:


cams_data["aartfaacmatch"] = np.min(np.abs((times_aartfaac - Time(list(cams_data["Time"].values))[:, np.newaxis]).to(u.s).value), axis=1) < 1


# In[40]:


cams_data[cams_data["aartfaacmatch"]].index


# In[41]:


len(cams_data.index[cams_data["aartfaacmatch"]])


# In[102]:


def showmean(data, header, i=0, step=60*u.s, central_time=None, ax=None, begin_chan=0, end_chan=-1):
    t0 = Time(header['MJD-OBS'], format='mjd') + header['CRVAL1'] * u.s
    delta_t = header['CDELT1'] * u.s
    ntimes = round((step/delta_t).si.value)
    if central_time is not None:
        begin_time_idx = max(int((central_time - t0).to(u.s) / delta_t - ntimes/2), 0)
    else:
        begin_time_idx = i * ntimes
    end_time_idx = begin_time_idx + ntimes
    mydata = data
    if len(mydata.shape) == 3:
        mydata = mydata[0]
    mydata = np.mean(mydata[begin_chan:end_chan, begin_time_idx:end_time_idx], axis=0)
    times = t0 + (np.arange(0, mydata.shape[-1]) + begin_time_idx) * delta_t
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(24,3))
    ax.plot(times.datetime, mydata)
    ax.set_aspect('auto')
    ax.set_xlim(np.min(times).datetime, np.max(times).datetime)
    ax.set_yticks([])
    ax.set_title("Mean over all channels")
    #if central_time is not None:
    #    ax.vlines(central_time.datetime, np.min(mydata), np.max(mydata))


# In[43]:


cams_peak_times = sorted([t for t in all_peak_times if np.min(np.abs((t - Time(list(cams_data["Time"].values))).to(u.s).value)) < 1])


# In[44]:


plt.close('all')


# In[45]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})


# In[68]:


for t in cams_peak_times[:1]:
    showmean(sap0.data, sap0.header, central_time=t, begin_chan=128, end_chan=256)
    show(sap0.data, sap0.header, 185, central_time=t)


# In[141]:


len(cams_data[np.logical_and(cams_data["hasmatch"], cams_data["Max-mV"]<1)])


# In[49]:


fig, ax = plt.subplots()
plt.close(fig)


# In[56]:


sap0.data.shape


# In[93]:


#df_to_plot = cams_data[np.logical_and(cams_data["hasmatch"], cams_data["Max-mV"]<1)]
df_to_plot = cams_data
for camspoint in tqdm(df_to_plot.iterrows(), total=len(df_to_plot)):
    t = camspoint[1]["astropytime"]
    data_497, header_497 = channelize("/data1/L792714/cs/L792714_SAP001_B000_S0_P000_bf.h5", nchan=2**17, start=t - 60*u.s, total=300, stokesi=True,
                                      frequency=49970000, bandwidth=1000, nof_samples=5624954880, nbin=1)
    data_499, header_499 = channelize("/data1/L792714/cs/L792714_SAP001_B000_S0_P000_bf.h5", nchan=2**17, start=t - 60*u.s, total=300, stokesi=True,
                                      frequency=49990000, bandwidth=1000, nof_samples=5624954880, nbin=1)
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, figsize=(12, 14))
    show2d(sap0.data, sap0.header, central_time=t, chan_nr=192, step=120*u.s, num_channels=128, ax=ax0, maxquantile=.9999, drawline=False);
    ax0.set_title(f"#{camspoint[0]}: {str(t)[:-4]}: Full subband")
    ax0.set_xticks([])
    for annotatepoint in df_to_plot[np.logical_and(cams_data['astropytime'] < t + 60 * u.s, cams_data['astropytime'] > t - 60 * u.s)].iterrows():
        #ax0.axvline(central_time.datetime, 0, 1, color='blue', linewidth=0.5)
        ax0.annotate(f"#{annotatepoint[0]}", (annotatepoint[1]["astropytime"].datetime, 0.08), xycoords=('data', 'axes fraction'),
                     color='white', fontsize=10)
    show2d(data_499, header_499, central_time=t, step=120*u.s, chan_nr=freq_to_channel(49.99*u.MHz, header_499), num_channels=101, ax=ax1);
    ax1.set_title(f"Ieper (49.990MHz)")
    ax1.set_xticks([])
    show2d(data_497, header_497, central_time=t, step=120*u.s, chan_nr=freq_to_channel(49.97*u.MHz, header_497), num_channels=101, ax=ax2);
    ax2.set_title(f"Dourbes (49.970MHz)")
    showmean(sap0.data, sap0.header, central_time=t, step=120*u.s, begin_chan=128, end_chan=256, ax=ax3)
    ax3.set_title("Mean over all channels, subband 256", pad=-12)
    showmean(sap0.data, sap0.header, central_time=t, step=120*u.s, begin_chan=0, end_chan=128, ax=ax4)
    ax4.set_title("Mean over all channels, subband 255", pad=-12)
    fig.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = .15, wspace = 0)
    fig.savefig(f"/home/dijkema/sap1/cams{camspoint[0]}.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

