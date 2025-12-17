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
    def download_kepler_lightcurves(
        md_df,
        out_dir,
        mission="Kepler",
        cadence="long",
        overwrite=False,
        max_targets=None,
        flatten=True,
        flatten_window_length=401,
        store_both_raw_and_flat=True,
        verbose=True,
    ):
        '''
        Download and save stitched light curves for each unique KIC in md_df.

        Saves each star as: out_dir/KIC<kepid>.npz

        By default saves:
          - time
          - flux_raw
          - flux_flat (if flatten=True and store_both_raw_and_flat=True)
          - mission, cadence (as metadata strings)

        Notes:
          - Prefers PDCSAP_FLUX, falls back to default flux column if needed.
          - Retries once if the cached file is corrupt, after purging cache.
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
            
            if verbose:
                print(f"[download] KIC {kepid}…")
            try:
                result = search_lightcurve(f"KIC {kepid}", mission=mission, cadence=cadence)

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
                    if verbose:
                        print("  -> using PDCSAP_FLUX (download_all + stitch)")
                except Exception as e_pdcsap:
                    if verbose:
                        print(f"  -> PDCSAP_FLUX not available ({e_pdcsap}), using default flux")
                    lcc = download_all_with_retry(flux_column=None)

                if lcc is None or len(lcc) == 0:
                    if verbose:
                        print("  -> download_all returned nothing")
                    continue

                lc = lcc.stitch().remove_nans()

                # save raw first
                time = lc.time.value # typically BKJD for Kepler
                flux_raw = lc.flux.value # PCDSAP

                flux_flat = None
                if flatten:
                    try:
                        lc_flat = lc.flatten(window_length=flatten_window_length)  # conservative; adjust later if needed
                        flux_flat = lc_flat.flux.value
                    except Exception as e_flat:
                        if verbose:
                            print(f"  -> flatten failed ({e_flat}); saving only raw flux")


                # save
                save_kwargs = {
                    "time": np.asarray(time),
                    "flux_raw": np.asarray(flux_raw),
                    "mission": np.array(mission),
                    "cadence": np.array(cadence),
                }
                if store_both_raw_and_flat and (flux_flat is not None):
                    save_kwargs["flux_flat"] = np.asarray(flux_flat)
                elif (not store_both_raw_and_flat) and (flux_flat is not None):
                    # If user wants only one, overwrite raw with flat (optional behavior)
                    save_kwargs["flux_raw"] = np.asarray(flux_flat)

                np.savez_compressed(filename, **save_kwargs)

                if verbose:
                    n = len(time)
                    keys = list(save_kwargs.keys())
                    print(f"  -> saved {filename} (N={n}, keys={keys})")

            except Exception as e:
                if verbose:
                    print(f"  -> ERROR for KIC {kepid}: {e}")

def build_folded_lightcurve_dataset(
    m_dwarfs,
    npz_dir,
    N_samples=2048,
    label_col="disposition_label",
    period_col="koi_period",
    t0_col="koi_time0bk",
    use_flux="flux_flat",          # "flux_flat" or "flux_raw"
    phase_min=-0.5,
    phase_max=0.5,
    min_points_total=100,
    min_points_in_window=50,
    clip_sigma=None,               # e.g. 10.0 to clip outliers; None disables
    add_channel_dim=True,          # makes X shape (N, N_samples, 1) for CNNs
    verbose=False,
):
    '''
    Build a fixed-length *phase-folded* light curve dataset for a 1D CNN.

    Reads saved NPZ files (time + flux), normalizes flux, folds on (period, t0),
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

        # label
        try:
            label = int(row[label_col])
        except Exception:
            continue
        
        # ephemeris
        period = row.get(period_col, np.nan)
        t0 = row.get(t0_col, np.nan)

        if not (np.isfinite(period) and np.isfinite(t0)) or period <= 0:
            continue

        fname = os.path.join(npz_dir, f"KIC{kepid}.npz")
        if not os.path.exists(fname):
            continue

        data = np.load(fname, allow_pickle=True)
        if ("time" not in data.files) or (use_flux not in data.files):
            # fallback: if requested flux isn't present, try the other one
            alt = "flux_raw" if use_flux == "flux_flat" else "flux_flat"
            if ("time" not in data.files) or (alt not in data.files):
                continue
            flux_key = alt
        else:
            flux_key = use_flux

        t = np.asarray(data["time"])
        f = np.asarray(data[flux_key])

        mask = np.isfinite(t) & np.isfinite(f)
        t, f = t[mask], f[mask]
        if len(f) < min_points_total:
            continue

        # sort by time
        order = np.argsort(t)
        t, f = t[order], f[order]

        # normalize flux around 0
        med = np.median(f)
        if not np.isfinite(med) or med == 0:
            continue
        f = f / med - 1.0

        # optional outlier clip
        if clip_sigma is not None:
            mad = np.median(np.abs(f - np.median(f)))
            if np.isfinite(mad) and mad > 0:
                # approx sigma from MAD
                sigma = 1.4826 * mad
                lo = -clip_sigma * sigma
                hi = +clip_sigma * sigma
                f = np.clip(f, lo, hi)

        # ---- phase fold ----
        # koi_time0bk is in BKJD; your saved lc.time.value is usually BKJD too.
        phase = ((t - t0 + 0.5 * period) % period) / period - 0.5  # [-0.5, 0.5)

        # keep only desired phase range (usually [-0.5, 0.5])
        keep = (phase >= phase_min) & (phase <= phase_max)
        phase, f = phase[keep], f[keep]
        if len(f) < min_points_in_window:
            continue

        # sort by phase for interpolation
        order = np.argsort(phase)
        phase, f = phase[order], f[order]

        # If there are duplicate phases, keep first occurrence (simple + safe)
        # (More robust: binning + median. This is the minimal fix.)
        uniq_phase, uniq_idx = np.unique(phase, return_index=True)
        phase, f = phase[uniq_idx], f[uniq_idx]
        if len(f) < 10:
            continue

        # fixed phase grid
        phase_grid = np.linspace(phase_min, phase_max, N_samples)
        f_resampled = np.interp(phase_grid, phase, f).astype("float32")

        X.append(f_resampled)
        y.append(label)
        kepids_used.append(int(kepid))

        if verbose and (len(X) % 50 == 0):
            print(f"[build] kept {len(X)} objects so far…")

    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int64")

    if add_channel_dim:
        X = X[..., None]  # (N, N_samples, 1)

    return X, y, kepids_used
