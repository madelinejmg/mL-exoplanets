import numpy as np
import pandas as pd

class M_dwarfs:
    @staticmethod
    def selection(df, teff_range=(2400, 3900), max_radius=0.7, max_mass=0.6, verbose=True):
        '''
         Select M-dwarf stars from a Kepler KOI dataframe using stellar parameters.

        Aargs:
            df : pandas.DataFrame
                The input dataframe containing KOI data.
            teff_range : tuple (min_teff, max_teff)
                Effective temperature range for M-dwarfs (Kelvin).
            max_radius : float
                Maximum stellar radius (in solar radii).
            max_mass : float or None
                Maximum stellar mass (in solar masses). If None, mass is ignored.
            verbose : bool
                Whether to print status messages.

        Returns:
            df_m : pandas.DataFrame
                Filtered dataframe containing only M-dwarf stars.
            m_mask : pandas.Series (boolean)
                Boolean mask used to select M-dwarfs.
        '''
        required_cols = ["koi_steff", "koi_srad"]
        missing = [col for col in required_cols if col not in df.columns]

        # Handle missing required columns
        if missing:
            if verbose:
                print(f"WARNING: Missing column(s) {missing}; skipping M-dwarf filter.")
            return df.copy(), pd.Series([True] * len(df), index=df.index)

        # Temperature + radius mask
        m_mask = (
            df["koi_steff"].between(teff_range[0], teff_range[1], inclusive="both")
            & (df["koi_srad"] <= max_radius)
        )

        # Optional mass filter (only applied if max_mass is provided AND column exists)
        if max_mass is not None:
            if "koi_smass" in df.columns:
                m_mask &= df["koi_smass"] <= max_mass
            else:
                if verbose:
                    print("WARNING: Mass filtering requested but 'koi_smass' missing; skipping mass filter.")

        df_m = df[m_mask].copy()

        if verbose:
            print(f"M-dwarf selection applied: {df_m.shape[0]} rows kept out of {df.shape[0]}")

        return df_m #, m_mask


