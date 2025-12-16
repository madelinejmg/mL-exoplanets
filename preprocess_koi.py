import numpy as np
import pandas as pd
import os, sys

# Reading target list
#KOI_catalog = '/Volumes/G-DRIVE/ASTRGR6012/KOI_catalog.csv'
#KOI_targets = pd.read_csv(KOI_catalog, comment='#')
#print(len(KOI_targets))
#KOI_targets.columns

class KOI:
    @staticmethod
    def load_catalog(path: str, verbose: bool=True):
        '''
        Load the KOI cumulative catalog that was retrieved from the NASA Exoplanet Archive.

        Note: it does not hard-code file paths.
        '''
        path = '/Volumes/G-DRIVE/ASTRGR6012/KOI_catalog.csv'
        if verbose:
            print(f"[KOI] Loading KOI catalog from: {path}")
        df = pd.read_csv(path, comment='#')
        if verbose:
            print(f"[KOI] Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    
    @staticmethod
    def preprocess(df, verbose: bool=True):
        '''
        Preprocess the KOI table.
        
        It is a dataframe for machine learning, where it does the following:
         1. Drop unused columns
         2. Remove FALSE POSITIVE rows
         3. Create binary disposition_label (1: candidate; 0: confirmed)
         4. Handle NaN values
         5. One-hot encoding for koi_tce_delivname
        '''
        # Copying the original df
        df = df.copy()

        # Drop unused columns
        drop_cols = (['kepler_name', 'koi_pdisposition', 'koi_score'])
        existing_drop_cols = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing_drop_cols)

        if verbose:
            print(f"After dropping unused cols: {df.shape}")

        # Limit target values to CANDIDATE and CONFIRMED
        false_positive_rows = df.query("koi_disposition == 'FALSE POSITIVE'").index
        df = df.drop(false_positive_rows, axis=0).reset_index(drop=True)

        if verbose:
            print(f"After dropping FALSE POSITIVES: {df.shape}")

        # Transforming target column in binary data
        df['disposition_label'] = df['koi_disposition'].map({
            "CANDIDATE": 1,
            "CONFIRMED": 0})
        
        # Move the label column next to koi_disposition
        cols = list(df.columns)
        # remove the label column from wherever it is
        cols.remove('disposition_label')
        # find index of koi_disposition
        i = cols.index('koi_disposition')
        # insert label right after koi_disposition
        cols.insert(i + 1, 'disposition_label')
        # reorder dataframe
        df = df[cols]

        # Drop columns with all missing values
        err_drop_cols = (['koi_teq_err1', 'koi_teq_err2'])
        existing_err_cols = [c for c in err_drop_cols if c in df.columns]
        if existing_err_cols:
            df = df.drop(columns=existing_err_cols)

        # Fill remaining missing values
        if 'koi_tce_delivname' in df.columns:
            df['koi_tce_delivname'] = df['koi_tce_delivname'].fillna(df['koi_tce_delivname'].mode()[0])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for column in numeric_cols:
            if df[column].isna().any():
                df[column] = df[column].fillna(df[column].mean())

        # One-hot encode koi_tce_delivname column
        if 'koi_tce_delivname' in df.columns:
            delivname_dummies = pd.get_dummies(
                df['koi_tce_delivname'], prefix='delivname'
            )
            df = pd.concat([df, delivname_dummies], axis=1)
            df = df.drop('koi_tce_delivname', axis=1)

            # ensure 0/1 ints
            df[delivname_dummies.columns] = delivname_dummies.astype(int)

        if verbose:
            print(f"Final preprocessed shape: {df.shape}")

        return df
