
from os.path import exists
from os import makedirs
from os import listdir
from os.path import isfile, join
from pathlib import Path
import logging

import numpy as np
from functools import reduce
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import ascii


def main(plot_all=False):
    """
    Requires two input files with the following naming convention:

    1. photometry file    : cluster_name.dat
    2. APASS file         : cluster_name_apass.csv

    """

    col_IDs, V_min, V_max, eVmax, eBVmax, N_tol, outl_tol = params_input()

    # Generate output dir if it doesn't exist.
    if not exists('out'):
        makedirs('out')

    # Process all files inside 'in/' folder.
    clusters = get_files()
    if not clusters:
        print("No input cluster files found")

    data_all = [[] for _ in range(6)]
    mypath = Path().absolute()
    for final_phot in clusters:

        # Extract name of file without extension
        cl_name = final_phot[3:-4]

        # Set up logging module
        level = logging.INFO
        frmt = ' %(message)s'
        handlers = [
            logging.FileHandler(
                join(mypath, 'out', cl_name + '.log'), mode='w'),
            logging.StreamHandler()]
        logging.basicConfig(level=level, format=frmt, handlers=handlers)

        # Path to APASS region
        apass_reg = 'in/' + cl_name + "_apass.csv"

        logging.info("\nProcessing: {}...".format(cl_name))
        # Read cluster photometry.
        logging.info("\nRead final photometry")
        ra_p, dec_p, v_p, bv_p, b_p = photRead(
            final_phot, col_IDs, eVmax, eBVmax)

        # Read APASS data.
        logging.info("\nRead APASS file")
        apass = apassRead(apass_reg)

        # Center APASS and filter data
        logging.info("\nCenter APASS frame, filter data according to V range.")
        ra_apass, dec_apass, ra_iraf, dec_iraf, V_apass, B_apass, BV_apass,\
            v_iraf, b_iraf, bv_iraf =\
            centerFilter(ra_p, dec_p, v_p, b_p, bv_p, apass, V_max, V_min)

        # Find stars within match tolerance.
        logging.info("\nMatching stars...")
        x_a, y_a, x_i, y_i, V_a_f, B_a_f, BV_a_f, V_i_f, B_i_f, BV_i_f =\
            matchStars(
                ra_apass, dec_apass, ra_iraf, dec_iraf, V_apass, B_apass,
                BV_apass, v_iraf, b_iraf, bv_iraf, N_tol, outl_tol)

        if plot_all:
            # Store for plotting of all combined data
            for i, dd in enumerate([
                    V_a_f, B_a_f, BV_a_f, V_i_f, B_i_f, BV_i_f]):
                data_all[i] += list(dd)

        if not plot_all:
            if len(x_a) > 1:
                logging.info("\nEstimate mean/median differences")
                Vmed, Vmean, Vstd, Bmed, Bmean, Bstd, BVmed, BVmean, BVstd =\
                    diffsPhot(V_a_f, V_i_f, B_a_f, B_i_f, BV_a_f, BV_i_f)

                logging.info("\nPlotting...")
                makePlot(
                    cl_name, V_min, V_max, N_tol, ra_apass, dec_apass, V_apass,
                    ra_iraf, dec_iraf, v_iraf, x_a, y_a, x_i, y_i, V_a_f,
                    B_a_f, BV_a_f, V_i_f, B_i_f, BV_i_f, Vmed, Vmean, Vstd,
                    Bmed, Bmean, Bstd, BVmed, BVmean, BVstd)
            else:
                logging.info("\nERROR: no matches found. Halting.")

            logging.info("\nEnd")

    if plot_all:
        makePlotAll(data_all, N_tol)


def params_input():
    """
    Read input parameters from 'params_input.dat' file.
    """
    with open('params_input.dat', "r") as f_dat:
        # Iterate through each line in the file.
        for l, line in enumerate(f_dat):
            if not line.startswith("#") and line.strip() != '':
                reader = line.split()
                if reader[0] == 'CI':
                    col_IDs = reader[1:]
                if reader[0] == 'VM':
                    V_min, V_max, eVmax, eBVmax = list(map(float, reader[1:]))
                if reader[0] == 'TO':
                    N_tol = float(reader[1])
                if reader[0] == 'OM':
                    outl_tol = float(reader[1])

    return col_IDs, V_min, V_max, eVmax, eBVmax, N_tol, outl_tol


def get_files():
    '''
    Store the paths and names of all the input clusters stored in the
    input folder.
    '''

    cl_files = []
    for f in listdir('in/'):
        if isfile(join('in/', f)) and f.endswith('.dat'):
            cl_files.append(join('in/', f))

    return cl_files


def photRead(final_phot, col_IDs, eVmax, eBVmax):
    """
    Select a file with photometry to read and compare with APASS.
    """
    # Final calibrated photometry
    phot = ascii.read(final_phot, fill_values=('INDEF', np.nan))

    id_ra, id_dec, id_v, id_ev, id_bv, id_ebv = col_IDs
    ra, dec, v, bv, e_v, e_bv = phot[id_ra], phot[id_dec], phot[id_v],\
        phot[id_bv], phot[id_ev], phot[id_ebv]
    b = bv + v

    # Mask bad photometry
    msk0 = (v < 50.) & (bv < 50.)
    # Mask large errors
    msk1, msk2 = e_v < eVmax, e_bv < eBVmax
    msk = msk0 & msk1 & msk2
    ra, dec, v, bv, b = ra[msk], dec[msk], v[msk], bv[msk], b[msk]

    return ra, dec, v, bv, b


def apassRead(apass_reg):
    """
    Read APASS data.
    """
    apass = ascii.read(apass_reg, fill_values=(('NA', np.nan), ('', np.nan)))

    return apass


def centerFilter(ra_p, dec_p, v_p, b_p, bv_p, apass, mag_max, mag_min):
    """
    Center APASS frame, filter data according to V range.
    """
    # Center frame for APASS data with proper range.
    xmin, xmax, ymin, ymax = ra_p.min(), ra_p.max(), dec_p.min(), dec_p.max()
    ra_c, de_c = .5 * (xmin + xmax), .5 * (ymin + ymax)
    ra_l, de_l = .5 * (xmax - xmin), .5 * (ymax - ymin)
    logging.info("RA range : {:.1f} arcsec".format(ra_l * 3600))
    logging.info("DEC range: {:.1f} arcsec".format(de_l * 3600))

    # Filter APASS frame to match the observed frame.
    mask = [apass['radeg'] < ra_c + ra_l, ra_c - ra_l < apass['radeg'],
            apass['decdeg'] < de_c + de_l, apass['decdeg'] > de_c - de_l,
            mag_min < apass['Johnson_V (V)'], apass['Johnson_V (V)'] < mag_max]
    total_mask = reduce(np.logical_and, mask)
    ra_apass = apass['radeg'][total_mask]
    deg_apass = apass['decdeg'][total_mask]
    V_apass = apass['Johnson_V (V)'][total_mask]
    B_apass = apass['Johnson_B (B)'][total_mask]
    BV_apass = B_apass - V_apass
    ra_a, dec_a = ra_apass, deg_apass
    logging.info("Max APASS V: {:.1f}".format(max(V_apass)))

    # Filter observed data to the fixed magnitude range.
    mask = [mag_min < v_p, v_p < mag_max]  # , ev_p < .03
    mask = reduce(np.logical_and, mask)
    logging.info("Mag limit for IRAF: {}".format(mag_max))
    ra_i, dec_i, v_i = ra_p[mask], dec_p[mask], v_p[mask]
    b_i, bv_i = b_p[mask], bv_p[mask]

    logging.info("APASS stars: {}".format(len(ra_a)))
    logging.info("IRAF stars: {}".format(len(ra_i)))

    return ra_a, dec_a, ra_i, dec_i, V_apass, B_apass, BV_apass, v_i, b_i, bv_i


def closestStar(x_fr1, y_fr1, x_fr2, y_fr2):
    """
    For every star in fr1, find the closest star in fr2.

    Parameters
    ----------
    x_fr1 : list
       x coordinates for stars in the reference frame.
    y_fr1 : list
       y coordinates for stars in the reference frame.
    x_fr2 : list
       x coordinates for stars in the processed frame.
    y_fr2 : list
       y coordinates for stars in the processed frame.

    Returns
    -------
    min_dist_idx : numpy array
        Index to the processed star closest to the reference star, for each
        reference star:
        * fr2[min_dist_idx[i]]: closest star in fr2 to the ith star in fr1.
        Also the index of the minimum distance in dist[i], i.e.: distance to
        the closest processed star to the ith reference star:
        * dist[i][min_dist_idx[i]]: distance between these two stars.
    min_dists : list
        Minimum distance for each star in the reference frame to a star in the
        processed frame.

    Notes
    -----
    len(fr1) = len(dist) = len(min_dist_idx)

    """
    fr1 = np.array(list(zip(*[x_fr1, y_fr1])))
    fr2 = np.array(list(zip(*[x_fr2, y_fr2])))
    min_dists, min_dist_idx = cKDTree(fr2).query(fr1, 1)

    return min_dist_idx, min_dists


def matchStars(
    x_apass, y_apass, x_iraf, y_iraf, V_apass, B_apass, BV_apass, v_iraf,
        b_iraf, bv_iraf, N_tol, outl_tol):
    """
    """
    min_dist_idx, min_dists = closestStar(x_apass, y_apass, x_iraf, y_iraf)

    logging.info("Match tolerance: {} arcsec".format(N_tol))
    logging.info("Outlier tolerance: {} mag".format(outl_tol))
    rad = (1. / 3600) * N_tol
    x_a, y_a, x_i, y_i = [], [], [], []
    V_a_f, B_a_f, BV_a_f, V_i_f, B_i_f, BV_i_f = [], [], [], [], [], []
    for st1_i, st2_i in enumerate(min_dist_idx):
        d = min_dists[st1_i]
        if d < rad:  # and abs(V_apass[st1_i] - v_i[st2_i]) < .5:
            # print("St1, St2: d={:.5f}".format(d))
            # print(" ({:.2f}, {:.2f}) ; ({:.2f}, {:.2f})".format(
            #     x[st1_i], y[st1_i], x_i[st2_i], y_i[st2_i]))
            # print(" V_1={:.2f}, V_2={:.2f}".format(
            #     V_apass[st1_i], v_i[st2_i]))
            if abs(V_apass[st1_i] - v_iraf[st2_i]) > outl_tol:
                logging.info(
                    '  Outlier: {}, {}'.format(V_apass[st1_i], v_iraf[st2_i]))
            else:
                x_a.append(x_apass[st1_i])
                y_a.append(y_apass[st1_i])
                V_a_f.append(V_apass[st1_i])
                B_a_f.append(B_apass[st1_i])
                BV_a_f.append(BV_apass[st1_i])

                x_i.append(x_iraf[st2_i])
                y_i.append(y_iraf[st2_i])
                V_i_f.append(v_iraf[st2_i])
                B_i_f.append(b_iraf[st2_i])
                BV_i_f.append(bv_iraf[st2_i])

    logging.info("Matched stars: {}".format(len(x_a)))

    V_a_f, B_a_f, BV_a_f, V_i_f, B_i_f, BV_i_f =\
        np.array(V_a_f), np.array(B_a_f), np.array(BV_a_f),\
        np.array(V_i_f), np.array(B_i_f), np.array(BV_i_f)
    return x_a, y_a, x_i, y_i, V_a_f, B_a_f, BV_a_f, V_i_f, B_i_f, BV_i_f


def diffsPhot(V_a_f, V_i_f, B_a_f, B_i_f, BV_a_f, BV_i_f):
    """
    """
    Vmed, Vmean, Vstd = np.nanmedian(V_a_f - V_i_f),\
        np.nanmean(V_a_f - V_i_f), np.nanstd(V_a_f - V_i_f)

    Bmed, Bmean, Bstd = np.nanmedian(B_a_f - B_i_f),\
        np.nanmean(B_a_f - B_i_f), np.nanstd(B_a_f - B_i_f)

    BVmed, BVmean, BVstd = np.nanmedian(BV_a_f - BV_i_f),\
        np.nanmean(BV_a_f - BV_i_f), np.nanstd(BV_a_f - BV_i_f)

    logging.info("median (V_APASS-V_IRAF): {:.4f}".format(Vmed))
    logging.info("median (B_APASS-B_IRAF): {:.4f}".format(Bmed))
    logging.info("median (BV_APASS-BV_IRAF): {:.4f}".format(BVmed))

    logging.info(
        r"xxx {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {}".format(
        Vmean, Vstd, Bmean, Bstd, BVmean, BVstd, len(V_a_f)))

    return Vmed, Vmean, Vstd, Bmed, Bmean, Bstd, BVmed, BVmean, BVstd


def makePlot(
    f_id, V_min, V_max, N_tol, x_apass, y_apass, V_apass, x_iraf, y_iraf,
    v_iraf, x_a, y_a, x_i, y_i, V_a_f, B_a_f, BV_a_f, V_i_f, B_i_f,
        BV_i_f, Vmed, Vmean, Vstd, Bmed, Bmean, Bstd, BVmed, BVmean, BVstd):
    """
    """
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(12, 18)

    plt.subplot(gs[0:6, 0:6])
    plt.title(r"APASS ($N$={}, $rad_{{match}}$={} arcsec)".format(
        len(x_apass), N_tol))
    plt.gca().invert_xaxis()
    plt.xlabel("ra")
    plt.ylabel("dec")
    plt.scatter(x_apass, y_apass, s=star_size(V_apass), c='r')
    plt.scatter(x_i, y_i, s=star_size(V_i_f) * .5)

    plt.subplot(gs[0:6, 6:12])
    plt.title(r"IRAF (N={}, $N_{{matched}}$={})".format(
        len(x_iraf), len(x_a)))
    plt.gca().invert_xaxis()
    plt.xlabel("ra")
    plt.ylabel("dec")
    plt.scatter(x_iraf, y_iraf, s=star_size(v_iraf), c='r')
    plt.scatter(x_a, y_a, s=star_size(V_a_f) * .5)

    plt.subplot(gs[0:3, 12:18])
    plt.title("{:.1f} < V < {:.1f}".format(V_min, V_max))
    plt.xlabel(r"$V_{{APASS}}$")
    plt.ylabel(r"$V_{{IRAF}}$")
    plt.scatter(V_a_f, V_i_f, s=4)
    plt.plot([min(V_a_f), max(V_a_f)], [min(V_a_f), max(V_a_f)], c='r')
    plt.xlim(min(V_a_f), max(V_a_f))
    plt.ylim(min(V_a_f), max(V_a_f))

    plt.subplot(gs[3:6, 12:18])
    plt.xlabel(r"$B_{{APASS}}$")
    plt.ylabel(r"$B_{{IRAF}}$")
    plt.scatter(B_a_f, B_i_f, s=4)
    plt.plot([min(B_a_f), max(B_a_f)], [min(B_a_f), max(B_a_f)], c='r')
    plt.xlim(min(B_a_f), max(B_a_f))
    plt.ylim(min(B_a_f), max(B_a_f))

    plt.subplot(gs[6:12, 0:6])
    plt.ylim(-.5, .5)
    plt.title(
        r"$\Delta V_{{mean}}=${:.4f}$\pm${:.4f}, ".format(Vmean, Vstd) +
        r"$\Delta V_{{median}}=${:.4f}".format(Vmed), fontsize=12)
    plt.xlabel(r"$V_{{APASS}}$")
    plt.ylabel(r"$V_{{APASS}}-V_{{IRAF}}$")
    plt.scatter(V_a_f, V_a_f - V_i_f, s=4)
    plt.axhline(y=Vmed, c='r')
    plt.axhline(y=Vmean, ls='--', c='g')

    plt.subplot(gs[6:12, 6:12])
    plt.ylim(-.5, .5)
    plt.title(
        r"$\Delta B_{{mean}}=${:.4f}$\pm${:.4f}, ".format(Bmean, Bstd) +
        r"$\Delta B_{{median}}=${:.4f}".format(Bmed), fontsize=12)
    plt.xlabel(r"$B_{{APASS}}$")
    plt.ylabel(r"$B_{{APASS}}-B_{{IRAF}}$")
    plt.scatter(B_a_f, B_a_f - B_i_f, s=4)
    plt.axhline(y=Bmed, c='r')
    plt.axhline(y=Bmean, ls='--', c='g')

    plt.subplot(gs[6:12, 12:18])
    plt.title(
        r"$\Delta BV_{{mean}}=${:.4f}$\pm${:.4f}, ".format(BVmean, BVstd) +
        r"$\Delta BV_{{median}}=${:.4f}".format(BVmed), fontsize=12)
    plt.scatter(BV_a_f, V_a_f, s=7, label="APASS")
    plt.scatter(BV_i_f, V_i_f, s=7, label="IRAF")
    plt.xlim(max(min(BV_a_f) - .3, -.3), max(BV_a_f) + .3)
    plt.ylim(min(V_a_f) - .5, max(V_a_f) + .25)
    plt.xlabel(r"$(B-V)$")
    plt.ylabel(r"$V$")
    plt.gca().invert_yaxis()
    plt.legend()

    fig.tight_layout()
    plt.savefig(
        'out/apass_' + f_id + '.png', dpi=300, bbox_inches='tight')


def makePlotAll(data_all, N_tol):
    """
    data_all = (V_apass, B_apass, BV_apass, V_iraf, B_iraf, BV_iraf)
    """
    data_all = np.array(data_all)
    V_apass, B_apass, BV_apass, V_iraf, B_iraf, BV_iraf = data_all

    plt.style.use('seaborn-darkgrid')
    plt.set_cmap('viridis')
    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(4, 4)

    minmax = 1.

    plt.subplot(gs[0])
    # plt.ylim(-.5, .5)
    plt.xlabel(r"$V$")
    plt.ylabel(r"$V_{{APASS}}-V$")
    delta_V = V_apass - V_iraf
    msk = (-minmax < delta_V) & (delta_V < minmax)
    Vmean, Vstd = np.nanmean(delta_V[msk]), np.nanstd(delta_V[msk])
    plt.title("N={}, mask=(-{}, {})".format(len(delta_V[msk]), minmax, minmax))
    plt.scatter(V_iraf[msk], delta_V[msk], s=8, c=BV_iraf[msk])
    plt.axhline(
        y=Vmean, ls='--', c='r',
        label=r"$\Delta V_{{mean}}=${:.4f}$\pm${:.4f}".format(Vmean, Vstd))
    plt.axhline(
        y=np.nanmedian(delta_V[msk]), ls=':', c='k',
        label="Median = {:.4f}".format(np.nanmedian(delta_V[msk])))
    plt.legend(fontsize=12)

    plt.subplot(gs[1])
    plt.xlabel(r"$B}$")
    plt.ylabel(r"$B_{{APASS}}-B$")
    delta_B = B_apass - B_iraf
    msk = (-minmax < delta_B) & (delta_B < minmax)
    Bmean, Bstd = np.nanmean(delta_B[msk]), np.nanstd(delta_B[msk])
    plt.title("N={}, mask=(-{}, {})".format(len(delta_B[msk]), minmax, minmax))
    plt.scatter(B_iraf[msk], delta_B[msk], s=8, c=BV_iraf[msk])
    plt.axhline(
        y=Bmean, ls='--', c='r',
        label=r"$\Delta B_{{mean}}=${:.4f}$\pm${:.4f}".format(Bmean, Bstd))
    plt.axhline(
        y=np.nanmedian(delta_B[msk]), ls=':', c='k',
        label="Median = {:.4f}".format(np.nanmedian(delta_B[msk])))
    plt.legend(fontsize=12)

    ax = plt.subplot(gs[2])
    plt.xlabel(r"$V$")
    plt.ylabel(r"$BV_{{APASS}}-BV$")
    delta_BV = BV_apass - BV_iraf
    msk = (-minmax < delta_BV) & (delta_BV < minmax)
    BVmean, BVstd = np.nanmean(delta_BV[msk]), np.nanstd(delta_BV[msk])
    plt.title(
        "N={}, mask=(-{}, {})".format(len(delta_BV[msk]), minmax, minmax))
    im = plt.scatter(V_iraf[msk], delta_BV[msk], s=8, c=BV_iraf[msk])
    plt.axhline(
        y=BVmean, ls='--', c='r',
        label=r"$\Delta BV_{{mean}}=${:.4f}$\pm${:.4f}".format(BVmean, BVstd))
    plt.axhline(
        y=np.nanmedian(delta_BV[msk]), ls=':', c='k',
        label="Median = {:.4f}".format(np.nanmedian(delta_BV[msk])))
    plt.legend(fontsize=12)

    print(N_tol, Vmean, Bmean, BVmean)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('(B-V)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    plt.savefig('out/apass_all.png', dpi=300, bbox_inches='tight')


def star_size(mag, N=None, min_m=None):
    '''
    Convert magnitudes into intensities and define sizes of stars in
    finding chart.
    '''
    # Scale factor.
    if N is None:
        N = len(mag)
    if min_m is None:
        min_m = np.nanmin(mag)
        # print("min mag used: {}".format(min_m))
    factor = 500. * (1 - 1 / (1 + 150 / N ** 0.85))
    return 0.1 + factor * 10 ** ((np.array(mag) - min_m) / -2.5)


if __name__ == '__main__':
    main()
