#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from astropy.io import ascii

if __name__ == "__main__":
    d = ascii.read("aartfaac_cams_perseids2020.ecsv")

    fig, ax = plt.subplots()

        
    x = np.sort(d["Hbeg"])
    y = np.linspace(0, 1, len(x))
    ax.plot(x, y, "k-", label="Optical begin")
    print(np.mean(x), np.std(x))

    x = np.sort(d["Hend"])
    y = np.linspace(0, 1, len(x))
    ax.plot(x, y, "k:", label="Optical end")
    print(np.mean(x), np.std(x))

    c = d["Hbeg_radio"] > 0
    x = np.sort(d["Hbeg_radio"][c])
    y = np.linspace(0, 1, len(x))
    ax.plot(x, y, "r-", label="Radio begin")
    print(np.mean(x), np.std(x))
    
    c = d["Hend_radio"] > 0
    x = np.sort(d["Hend_radio"][c])
    y = np.linspace(0, 1, len(x))
    ax.plot(x, y, "r:", label="Radio end")
    print(np.mean(x), np.std(x))

    ax.legend()
    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel("Cumulative fraction")
    
    plt.tight_layout()
    plt.savefig("altitudes.png", bbox_inches="tight")


    c = d["AARTFAAC_duration"] > 0
        

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 8))

    ax1.plot(d["Int-mV"][c], d["AARTFAAC_duration"][c], "k.")
    ax1.set_xlim(2, -11)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.text(1.7, 300, "a")
    
    ax2.plot(d["Int-mV"][c], d["Hbeg"][c], "k.", label="Optical begin")
    ax2.plot(d["Int-mV"][c], d["Hbeg_radio"][c], "r.", label="Radio begin")
    ax2.legend(loc="lower right")
    ax2.text(1.7, 132, "b")
             
    ax3.plot(d["Int-mV"][c], d["Hend"][c], "k.", label="Optical end")
    ax3.plot(d["Int-mV"][c], d["Hend_radio"][c], "r.", label="Radio end")
    ax3.legend(loc="upper right")
    ax3.text(1.7, 107, "c")
    
    ax3.set_xlabel("Optical magnitude")
    ax1.set_ylabel("Radio duration (s)")
    ax2.set_ylabel("Altitude (km)")
    ax3.set_ylabel("Altitude (km)")

    plt.tight_layout()
    plt.savefig("magnitude.pdf", bbox_inches="tight")

