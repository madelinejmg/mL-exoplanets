import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.metrics import make_scorer, roc_auc_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from preprocess_koi import KOI
from m_dwarf_selection import M_dwarfs

class KOIPipeline:
    @staticmethod
    def run_data(path):
        df_raw = KOI.load_catalog(path)
        print('*********************************')
        df_clean = KOI.preprocess(df_raw)
        print('*********************************')
        df_md = M_dwarfs.selection(df_clean)
        return {"raw": df_raw,
                "clean": df_clean,
                "m_dwarfs": df_md}
    
    @staticmethod
    def model_features(df, verbose=True):
        '''
        Splitting the m_dwarfs dataset into two parts:

        Args:
            X : numpy.ndarray
                independent variables, feature matrix
            y : numpy.ndarray
                dependent variable ('deposition_label'), label vector
            feature_names : list
                List of feature column names used in X.
        '''
        if "disposition_label" not in df.columns:
            raise ValueError("ERROR: 'disposition_label' not found. Did you run preprocess()?")
        
        y = df["disposition_label"].values

        # feature extraction
        drop_cols = ["koi_disposition",      # text label
                    "disposition_label",]   # target
        
        # Remove non-numeric columns & drop irrelevant ones
        numeric_df = df.drop(columns=drop_cols, errors="ignore")
        numeric_df = numeric_df.select_dtypes(include=["number"])

        feature_names = numeric_df.columns.tolist()
        X = numeric_df.values

        if verbose:
            print(f"[build] Feature matrix shape: {X.shape}")
            print(f"[build] Label vector shape:   {y.shape}")
            print(f"[build] Num features: {len(feature_names)}")

        return X, y, feature_names
    
    @staticmethod
    def validate_model(model, X, y, cv=5, verbose=True):
        '''
        Perform k-fold cross-validation on (X, y) using the given model.

        Args:
            model : sklearn estimator
                The classifier (e.g., RandomForestClassifier instance).
            X : numpy.ndarray
                Feature matrix.
            y : numpy.ndarray
                Label vector.
            cv : int
                Number of folds for k-fold cross-validation.
            verbose : bool
                Whether to print validation results.

        Returns:
            scores : dict
                Dictionary with mean/std accuracy and AUC, plus raw fold scores.
        '''
        # AUC scorer (needs probabilities, not just class predictions)
        auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

        # Accuracy scores across folds
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        # AUC scores across folds
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring=auc_scorer)

        scores = {
            "accuracy_mean": acc_scores.mean(),
            "accuracy_std": acc_scores.std(),
            "auc_mean": auc_scores.mean(),
            "auc_std": auc_scores.std(),
            "acc_scores": acc_scores,
            "auc_scores": auc_scores,
        }

        if verbose:
            print(f"[validation] {cv}-fold cross-validation")
            print(f"  Accuracy: {scores['accuracy_mean']:.3f} ± {scores['accuracy_std']:.3f}")
            print(f"  AUC:      {scores['auc_mean']:.3f} ± {scores['auc_std']:.3f}")

        return scores

    def split_data(self, X, y, random_state=42, verbose=True, plot_classes=False, min_class_count_warning=3):
        '''
        Split the data into train (80%), validation (10%), and test (10%).

        Args:
            X : numpy.ndarray
                Feature matrix.
            y : numpy.ndarray
                Label vector.
            test_size : float
                Fraction of the data to reserve for testing.
            verbose : bool
                Whether to print split information.

        Returns:
            X_train, X_test, y_train, y_test : arrays
                The split datasets.
        '''
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=random_state,
            stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=random_state,
            stratify=y_temp
        )

        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.X_test,  self.y_test  = X_test,  y_test

        if verbose:
            print("[split] Train:", X_train.shape[0])
            print("[split] Val:  ", X_val.shape[0])
            print("[split] Test: ", X_test.shape[0])

        def balance(arr):
            '''Return counts of each class in a clean format.'''
            classes, counts = np.unique(arr, return_counts=True)
            return dict(zip(classes, counts))
        
        train_bal = balance(y_train)
        val_bal = balance(y_val)
        test_bal = balance(y_test)

        if verbose:
            print("[balance] Train:", train_bal)
            print("[balance] Val:  ", val_bal)
            print("[balance] Test: ", test_bal)

                
            for split_name, bal in [("Train", train_bal),
                                    ("Val", val_bal),
                                    ("Test", test_bal)]:
                for cls, cnt in bal.items():
                    if cnt < min_class_count_warning:
                        print(
                            f"WARNING: Only {cnt} examples of class {cls} in {split_name} set "
                            f"(recommended minimum: {min_class_count_warning})."
                        )

        
        if plot_classes:
            splits = [("Train", y_train), ("Val", y_val), ("Test", y_test)]
            n_splits = len(splits)

            fig, axes = plt.subplots(1, n_splits, figsize=(4 * n_splits, 4))
            if n_splits == 1:
                axes = [axes]

            for ax, (name, y_split) in zip(axes, splits):
                classes, counts = np.unique(y_split, return_counts=True)
                ax.bar(classes, counts)
                ax.set_title(f"{name} Class Counts")
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")

            plt.tight_layout()
            plt.show()

        return X_train, X_val, X_test, y_train, y_val, y_test
        
    @staticmethod
    def plot_class_counts(y_train, y_val=None, y_test=None):
        '''
        Plot bar charts of class counts for train/val/test splits.
        '''
        splits = [("Train", y_train)]
        if y_val is not None:
            splits.append(("Val", y_val))
        if y_test is not None:
            splits.append(("Test", y_test))

        n_splits = len(splits)
        fig, axes = plt.subplots(1, n_splits, figsize=(4 * n_splits, 4), squeeze=False)

        for ax, (name, y_split) in zip(axes[0], splits):
            classes, counts = np.unique(y_split, return_counts=True)
            ax.bar(classes, counts)
            ax.set_title(f"{name} class counts")
            ax.set_xlabel("Class label")
            ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def group_kfold_indices(groups, n_splits=5):
        '''
        Generate (train_idx, val_idx) pairs for GroupKFold given group labels.

        Note: a static helper that yields train/val indices for GroupKFold. 
        This doesn not do the CNN training; it just gives the splits correctly.
        
        Args:
            groups : array-like
                Group labels (e.g., kepid per TCE or light curve).
            n_splits : int
                Number of folds.

        Returns:
            folds : list of (train_idx, val_idx)
                Each element is a pair of index arrays for a fold.
        '''
        groups = np.asarray(groups)
        gkf = GroupKFold(n_splits=n_splits)

        idx = np.arange(len(groups))
        return [(train_idx, val_idx) for train_idx, val_idx in gkf.split(idx, groups=groups)]
    
    @staticmethod
    def compute_class_weights(y):
        '''
        Compute inverse-frequency class weights for binary labels (0/1).
        '''
        y = np.asarray(y)
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        weights = {}
        for cls, cnt in zip(unique, counts):
            weights[int(cls)] = total / (2.0 * cnt)
        return weights
    
    @staticmethod
    def build_cnn(input_length):
        '''
        Build a 1D CNN for binary classification on light curves.
        input_length: number of time samples (e.g., 2048).
        '''
        model = models.Sequential([
            layers.Input(shape=(input_length, 1)),

            layers.Conv1D(32, kernel_size=7, padding="same", activation="relu"),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(64, kernel_size=7, padding="same", activation="relu"),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(128, kernel_size=7, padding="same", activation="relu"),
            layers.GlobalAveragePooling1D(),

            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),

            layers.Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ]
        )

        return model

    def train_cnn_groupkfold(
            self,
            X_lc,
            y,
            groups,
            n_splits=5,
            epochs=40,
            batch_size=8,
            patience=5,
            threshold=0.5,
        ):
            '''
            GroupKFold CV for your CNN.
            Groups should be kepids_used aligned with rows of X_lc/y.
            Uses KOIPipeline.split_data for train/val/test, then reshapes for CNN.
            '''
            X_lc = np.asarray(X_lc, dtype="float32")
            y = np.asarray(y, dtype="int64")
            groups = np.asarray(groups)

            folds = self.group_kfold_indices(groups, n_splits=n_splits)

            fold_stats = []
            all_probs = np.full(len(y), np.nan, dtype="float32")  # optional: store OOF probs

            for k, (train_idx, val_idx) in enumerate(folds, start=1):
                print(f"\n===== GroupKFold {k}/{n_splits} =====")

                X_train, y_train = X_lc[train_idx], y[train_idx]
                X_val,   y_val   = X_lc[val_idx],   y[val_idx]

                # reshape
                X_train_cnn = X_train[..., np.newaxis]
                X_val_cnn   = X_val[..., np.newaxis]

                # weights from train fold only
                class_weight = self.compute_class_weights(y_train)
                print("[balance] Train:", dict(zip(*np.unique(y_train, return_counts=True))))
                print("[balance] Val:  ", dict(zip(*np.unique(y_val, return_counts=True))))
                print("[cnn] class_weight:", class_weight)

                # fresh model each fold
                tf.keras.backend.clear_session()
                model = self.build_cnn(input_length=X_train_cnn.shape[1])

                es = callbacks.EarlyStopping(
                    monitor="val_auc",
                    mode="max",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1,
                )

                history = model.fit(
                    X_train_cnn, y_train,
                    validation_data=(X_val_cnn, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_weight,
                    callbacks=[es],
                    verbose=1,
                )

                # evaluate on fold val
                y_prob = model.predict(X_val_cnn).ravel()
                y_pred = (y_prob >= threshold).astype(int)
                all_probs[val_idx] = y_prob  # out-of-fold probabilities

                try:
                    auc = roc_auc_score(y_val, y_prob)
                except ValueError:
                    auc = np.nan

                print("Fold AUC:", auc)
                print(confusion_matrix(y_val, y_pred))

                fold_stats.append({"fold": k, "auc": auc, "history": history.history})

            # summary across folds
            aucs = np.array([d["auc"] for d in fold_stats], dtype=float)
            print("\n===== CV Summary =====")
            print("AUCs:", aucs)
            print("Mean AUC:", np.nanmean(aucs), "Std:", np.nanstd(aucs))

            # overall OOF AUC 
            try:
                oof_auc = roc_auc_score(y[~np.isnan(all_probs)], all_probs[~np.isnan(all_probs)])
                print("OOF ROC AUC:", oof_auc)
            except ValueError:
                print("OOF ROC AUC could not be computed.")

            return fold_stats, all_probs

    @staticmethod
    def evaluate_cnn(model, X_test_cnn, y_test, threshold=0.5):
        '''
        Evaluate the trained CNN on the test set:
        prints classification report, confusion matrix, ROC AUC.
        '''
        y_prob = model.predict(X_test_cnn).ravel()
        y_pred = (y_prob >= threshold).astype(int)

        print("\n=== CNN Test Performance (threshold = {:.2f}) ===".format(threshold))
        print(classification_report(y_test, y_pred, digits=3))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        try:
            auc = roc_auc_score(y_test, y_prob)
            print("ROC AUC: {:.3f}".format(auc))
        except ValueError:
            print("ROC AUC could not be computed (only one class in y_test).")

        return y_pred, y_prob

    @staticmethod
    def plot_training_history(history):
        '''
        Plot training/validation loss and accuracy from a Keras History object.
        '''
        hist = history.history

        plt.figure(figsize=(10, 4))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(hist["loss"], label="train")
        plt.plot(hist["val_loss"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()

        # Accuracy (if available)
        if "accuracy" in hist:
            plt.subplot(1, 2, 2)
            plt.plot(hist["accuracy"], label="train")
            plt.plot(hist["val_accuracy"], label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy")
            plt.legend()

        plt.tight_layout()
        plt.show()
     

    def train_model(self, model=None):
        pass

