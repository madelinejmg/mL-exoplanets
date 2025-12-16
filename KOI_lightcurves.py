import os
import numpy as np
from lightkurve import search_lightcurve

import shutil
from pathlib import Path

class LightKurve_Fetcher:
    @staticmethod
    def _purge_corrupt_kic_cache(kepid, mission="Kepler"):
        '''
        Remove cached Lightkurve/MAST downloads for a given KIC/mission.
        This is safe: it only deletes cached FITS, not your saved .npz files.
        '''
        kepid_int = int(kepid)
        if mission.lower() == "kepler":
            # Kepler cache uses kplr00######## naming
            kplr = f"kplr{kepid_int:09d}"
            base = Path.home() / ".lightkurve" / "cache" / "mastDownload" / "Kepler"
            if base.exists():
                for p in base.glob(f"{kplr}_lc_*"):
                    shutil.rmtree(p, ignore_errors=True)
        else:
            # Generic fallback: wipe the mission cache directory if needed
            base = Path.home() / ".lightkurve" / "cache" / "mastDownload" / mission
            if base.exists():
                shutil.rmtree(base, ignore_errors=True)

    @staticmethod
    def download_Kepler_lightcurves(md_df, out_dir, mission="Kepler", overwrite=False, max_targets=None):
        '''
        Download and save stitched light curves for each unique KIC in md_df.

        Saves each star as: out_dir/KIC<kepid>.npz  with 'time' and 'flux' arrays.
        '''
        os.makedirs(out_dir, exist_ok=True)

        kepids = md_df["kepid"].unique()
        if max_targets is not None:
            kepids = kepids[:max_targets]

        for kepid in kepids:
            filename = os.path.join(out_dir, f"KIC{kepid}.npz")

            if (not overwrite) and os.path.exists(filename):
                print(f"[skip] {filename} already exists")
                continue

            print(f"[download] KIC {kepid}…")
            try:
                result = search_lightcurve(f"KIC {kepid}", mission=mission, cadence="long")

                if len(result) == 0:
                    print(f"  -> no light curves found")
                    continue

                # Helper: try download_all (optionally PDCSAP), and retry once if cache is corrupt
                def download_all_with_retry(flux_column=None):
                    for attempt in (1, 2):
                        try:
                            if flux_column is None:
                                return result.download_all()
                            else:
                                return result.download_all(flux_column=flux_column)
                        except Exception as e:
                            msg = str(e)
                            corrupt = ("file may be corrupt" in msg) or ("Not recognized as a supported data product" in msg)
                            if corrupt and attempt == 1:
                                print("  -> cache looks corrupt; purging and retrying once")
                                LightKurve_Fetcher._purge_corrupt_kic_cache(kepid, mission=mission)
                                continue
                            raise  # re-raise on 2nd failure or non-corrupt errors

                # Ask Lightkurve directly for PDCSAP as the flux column; Prefer PDCSAP, but stitch *all* available segments
                try:
                    lcc = download_all_with_retry(flux_column="pdcsap_flux")
                    print("  -> using PDCSAP_FLUX (download_all + stitch)")
                except Exception as e_pdcsap:
                    print(f"  -> PDCSAP_FLUX not available ({e_pdcsap}), using default flux")
                    lc = download_all_with_retry(flux_column=None)

                if lcc is None or len(lcc) == 0:
                    print("  -> download_all returned nothing")
                    continue

                lc = lcc.stitch().remove_nans()
                try:
                    lc = lc.flatten(window_length=401)  # conservative; adjust later if needed
                except Exception as e_flat:
                    print(f"  -> flatten failed ({e_flat}); saving unflattened light curve")

                time = lc.time.value # in days
                flux = lc.flux.value # PCDSAP

                np.savez_compressed(filename, time=time, flux=flux)
                print(f"  -> saved {filename} (N={len(time)})")

            except Exception as e:
                print(f"  -> ERROR for KIC {kepid}: {e}")

def build_folded_lightcurve_dataset(
    m_dwarfs,
    npz_dir,
    N_samples=2048,
    label_col="disposition_label",
    period_col="koi_period",
    t0_col="koi_time0bk",
    phase_min=-0.5,
    phase_max=0.5,
    max_phase_span=0.5,   # keeps [-0.5, 0.5] by default
):
    '''
    Build a fixed-length *phase-folded* light curve dataset for a 1D CNN.

    Reads saved NPZ files (time, flux), normalizes flux, folds on (period, t0),
    and interpolates flux onto a fixed phase grid of length N_samples.

    Returns:
        X: (N_objects, N_samples) float32
        y: (N_objects,) int64
        kepids_used: list
    '''
    X, y, kepids_used = [], [], []

    # one example per star
    for kepid, grp in m_dwarfs.groupby("kepid"):
        row = grp.iloc[0]
        label = int(row[label_col])

        period = row[period_col]
        t0 = row[t0_col]

        if not (np.isfinite(period) and np.isfinite(t0)) or period <= 0:
            continue

        fname = os.path.join(npz_dir, f"KIC{kepid}.npz")
        if not os.path.exists(fname):
            continue

        data = np.load(fname)
        t = data["time"]
        f = data["flux"]

        mask = np.isfinite(t) & np.isfinite(f)
        t, f = t[mask], f[mask]
        if len(f) < 100:
            continue

        # sort by time
        order = np.argsort(t)
        t, f = t[order], f[order]

        # normalize flux around 0
        med = np.median(f)
        if not np.isfinite(med) or med == 0:
            continue
        f = f / med - 1.0

        # ---- phase fold ----
        # koi_time0bk is in BKJD; your saved lc.time.value is usually BKJD too.
        phase = ((t - t0 + 0.5 * period) % period) / period - 0.5  # [-0.5, 0.5)

        # keep only desired phase range (usually [-0.5, 0.5])
        keep = (phase >= phase_min) & (phase <= phase_max)
        phase, f = phase[keep], f[keep]
        if len(f) < 50:
            continue

        # sort by phase for interpolation
        order = np.argsort(phase)
        phase, f = phase[order], f[order]

        # fixed phase grid
        phase_grid = np.linspace(phase_min, phase_max, N_samples)
        f_resampled = np.interp(phase_grid, phase, f)

        X.append(f_resampled.astype("float32"))
        y.append(label)
        kepids_used.append(kepid)

    return np.array(X, dtype="float32"), np.array(y, dtype="int64"), kepids_used
