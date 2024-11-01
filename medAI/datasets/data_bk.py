import re
import os
from glob import glob
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from skimage.transform import resize

from sklearn.model_selection import train_test_split

DATA_DIR_ROOT_MAIN = "/projects/bk_pca/BK_UBC_CORES/"
DATA_DIR_PATCH_ROOT = (
    "/ssd005/projects/exactvu_pca/bk_ubc/patches/UBC/patch_48x48_str32_avg/"
)

DATA_CORES_UBC = "/projects/bk_pca/BK_UBC_CORES/"
DATA_CORES_QUEENS = "/projects/bk_pca/BK_QUEENS_CORES/"


def get_patch_labels():
    patients = {}
    for patient in os.listdir(DATA_DIR_PATCH_ROOT):
        labels = []
        invs = []
        patient_id = patient.replace("Patient", "")
        try:
            for core in os.listdir(
                DATA_DIR_PATCH_ROOT + str(patient) + "/patches_rf_core"
            ):
                labels.append(1 if "cancer" in core else 0)
                inv = re.findall("_inv([\d.[0-9]+)", core)
                invs.append(float(inv[0]) if inv else 0)
            patients[patient_id] = (labels, invs)
        except FileNotFoundError:
            continue
    return patients


def build_label_table():
    df = pd.DataFrame()
    for key, value in get_patch_labels().items():
        labels, invs = value
        for i in range(len(labels)):
            label, inv = labels[i], invs[i]
            row = pd.DataFrame(
                {
                    "core_id": [f"{key}.{i}"],
                    "patient_id": [key],
                    "label": [label],
                    "inv": [inv],
                }
            )
            df = pd.concat([df, row], ignore_index=True)
    return df

def extract_info_json(info_array, df, _pandas_idx=0):
    import json

    dicts = [df] if df.shape[0] > 0 else []
    for info in info_array:
        data = json.load(open(info))
        center = re.findall("BK_(\w+)_CORES", info)[0]
        core_idx = re.findall("pat(\d+)_cor(\d+)", info)[0]
        core_id = f'{int(core_idx[0]):04d}.{core_idx[1]}'
        entry = {
            "core_id": f'{center.lower()}_{core_id}', #core_id,
            "center": center,
            "patient_id": f'{center.lower()}_{int(core_idx[0]):04d}', # core_id.split(".")[0], 
            "inv": data["Involvement"] / 100,
            "pathology": data["Pathology"],
            "label": 1 if data["Pathology"] == "Adenocarcinoma" else 0,
            "filetemplate": info.replace("_info.json", ""),
        }
        dicts.append(pd.DataFrame(entry, index=[_pandas_idx]))
        _pandas_idx += 1

    df = pd.concat(dicts)
    return df

def make_table(ubc=True, queens=True):
    df = pd.DataFrame(columns=["core_id", "patient_id", "inv", "pathology", "label"])
    ubc_cores = os.listdir(DATA_CORES_UBC)
    queens_cores = os.listdir(DATA_CORES_QUEENS)

    if ubc:
        ubc_info = [
            DATA_CORES_UBC + filename
            for filename in ubc_cores
            if filename.endswith(".json")
        ]
        df = extract_info_json(ubc_info, df)
        df.inv = df.inv.apply(lambda x: x*100) # fix bug
    if queens:
        queens_info = [
            DATA_CORES_QUEENS + filename
            for filename in queens_cores
            if filename.endswith(".json")
        ]
        df = extract_info_json(queens_info, df, df.shape[0] - 1)
    return df

def select_patients(all_files, patient_ids):
    patient_files = []
    for file in all_files:
        patient = int(re.findall("/Patient(\d+)/", file)[0])
        if patient in patient_ids:
            patient_files.append(file)
    return patient_files

def split_centerwise(inv_threshold=None, by_patients=None):
    table = make_table(ubc=True, queens=True)
    if inv_threshold:
        table = table[(table["inv"] >= inv_threshold) | (table["label"] == 0)]

    ubc_table = table[table.center == "UBC"]
    ubc_patients = ubc_table.drop_duplicates(subset=["patient_id"])
    queens_table = table[table.center == "QUEENS"]
    queens_patients = queens_table.drop_duplicates(subset=["patient_id"])

    val_pa, test_pa = train_test_split(
        ubc_patients, test_size=0.5, random_state=0, stratify=ubc_patients["label"]
    )

    train_idx = table.patient_id.isin(queens_table.patient_id)
    val_idx = table.patient_id.isin(val_pa.patient_id)
    test_idx = table.patient_id.isin(test_pa.patient_id)

    train_tab, val_tab, test_tab = table[train_idx], table[val_idx], table[test_idx]

    assert set(train_tab.patient_id) & set(val_tab.patient_id) == set()
    assert set(train_tab.patient_id) & set(test_tab.patient_id) == set()
    assert set(val_tab.patient_id) & set(test_tab.patient_id) == set()

    print(f"Train: {len(train_tab)}")
    print(f"Val: {len(val_tab)}")
    print(f"Test: {len(test_tab)}")

    print(train_tab.center.value_counts())

    return train_tab, val_tab, test_tab

def select_patients(patients, seed=0, oversample_cancer=False, undersample_benign=False, sampling_ratio=1):
    table = make_table(ubc=True, queens=True)
    patient_table = table.drop_duplicates(subset=["patient_id"])
    return table[table.patient_id.isin(patients)]

def split_patients(seed=0, oversample_cancer=False, undersample_benign=False, sampling_ratio=1):
    table = make_table(ubc=True, queens=True)
    patient_table = table.drop_duplicates(subset=["patient_id"])

    train_pa, val_pa = train_test_split(
        patient_table, test_size=0.3, random_state=seed, stratify=patient_table["label"]
    )

    val_pa, test_pa = train_test_split(
        val_pa, test_size=0.5, random_state=seed, stratify=val_pa["label"]
    )

    train_idx = table.patient_id.isin(train_pa.patient_id)
    val_idx = table.patient_id.isin(val_pa.patient_id)
    test_idx = table.patient_id.isin(test_pa.patient_id)

    train_tab, val_tab, test_tab = table[train_idx], table[val_idx], table[test_idx]

    num_benign = train_tab[train_tab.label == 0].shape[0]
    num_cancer = train_tab[train_tab.label == 1].shape[0]
    if oversample_cancer:
        num_resample = int(num_benign * sampling_ratio) - num_cancer
        train_tab = pd.concat([
            train_tab,
            train_tab[train_tab.label == 1].sample(num_resample, replace=True)
        ]).reset_index(drop=True)
    elif undersample_benign:
        num_resample = int(num_cancer / sampling_ratio)
        train_tab = pd.concat([
            train_tab[train_tab.label == 1],
            train_tab[train_tab.label == 0].sample(num_resample)
        ]).reset_index(drop=True)

    assert set(train_tab.patient_id) & set(val_tab.patient_id) == set()
    assert set(train_tab.patient_id) & set(test_tab.patient_id) == set()
    assert set(val_tab.patient_id) & set(test_tab.patient_id) == set()

    print(f"Train: {len(train_tab)}")
    print(f"Val: {len(val_tab)}")
    print(f"Test: {len(test_tab)}")

    print(train_tab.center.value_counts())

    return train_tab, val_tab, test_tab


def split_patients_kfold(
        fold_id, k=5, 
        seed=0, 
        centers=["UBC", "QUEENS"],
        oversample_cancer=False,
        undersample_benign=False,
        sampling_ratio=1
    ):
    table = make_table(ubc="UBC" in centers, 
                       queens="QUEENS" in centers)
    
    patient_table = table.drop_duplicates(subset=["patient_id"])

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(
        skf.split(patient_table, patient_table["label"])
    ):
        print(f"Fold {i}")
        print(f"Train: {len(train_idx)}")
        print(f"Test: {len(test_idx)}")
        if i == fold_id:
            train_pa, val_pa = (
                patient_table.iloc[train_idx],
                patient_table.iloc[test_idx],
            )
            val_pa, test_pa = train_test_split(
                val_pa, test_size=0.5, random_state=seed, stratify=val_pa["label"]
            )

            train_idx = table.patient_id.isin(train_pa.patient_id)
            val_idx = table.patient_id.isin(val_pa.patient_id)
            test_idx = table.patient_id.isin(test_pa.patient_id)

            train_tab, val_tab, test_tab = (
                table[train_idx],
                table[val_idx],
                table[test_idx],
            )

            assert set(train_tab.patient_id) & set(val_tab.patient_id) == set()
            assert set(train_tab.patient_id) & set(test_tab.patient_id) == set()
            assert set(val_tab.patient_id) & set(test_tab.patient_id) == set()

            num_benign = train_tab[train_tab.label == 0].shape[0]
            num_cancer = train_tab[train_tab.label == 1].shape[0]
            if oversample_cancer:
                num_resample = int(num_benign * sampling_ratio) - num_cancer
                train_tab = pd.concat([
                    train_tab,
                    train_tab[train_tab.label == 1].sample(num_resample, replace=True)
                ]).reset_index(drop=True)
            elif undersample_benign:
                num_resample = int(num_cancer / sampling_ratio)
                train_tab = pd.concat([
                    train_tab[train_tab.label == 1],
                    train_tab[train_tab.label == 0].sample(num_resample)
                ]).reset_index(drop=True)

            return train_tab, val_tab, test_tab


def make_bk_dataloaders(self_supervised=False):
    train_tab, val_tab, test_tab = split_patients()

    _BKPatchesDataset = (
        BKPatchLabeledDataset if not self_supervised else BKPatchUnlabeledDataset
    )
    transform = PatchTransform() if not self_supervised else PatchSSLTransform()
    train_ds = _BKPatchesDataset(
        DATA_DIR_PATCH_ROOT, patient_ids=train_tab.patient_id, transform=transform
    )
    val_ds = _BKPatchesDataset(
        DATA_DIR_PATCH_ROOT, patient_ids=val_tab.patient_id, transform=transform
    )
    test_ds = _BKPatchesDataset(
        DATA_DIR_PATCH_ROOT, patient_ids=test_tab.patient_id, transform=transform
    )

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_dl, val_dl, test_dl


def make_corewise_bk_dataloaders_old(
    batch_sz,
    im_sz=1024,
    style="avg_all",
    splitting="patients",
    fold=0,
    num_folds=5,
    seed=0,
    inv_threshold=None,
    centers=["UBC", "QUEENS"],
    oversampling=False,
    undersampling=False,
    sampling_ratio=1
):
    if splitting == "patients":
        train_tab, val_tab, test_tab = split_patients(seed=seed)
    elif splitting == "centers":
        train_tab, val_tab, test_tab = split_centerwise()
    elif splitting == "patients_kfold":
        train_tab, val_tab, test_tab = split_patients_kfold(
            fold, k=num_folds, 
            seed=seed,
            centers=centers,
            oversample_cancer=oversampling, 
            undersample_benign=undersampling, 
            sampling_ratio=sampling_ratio
        )
    else:
        print("Invalid splitting method. Using patients.")
        train_tab, val_tab, test_tab = split_patients(seed=seed)

    if inv_threshold:
        assert type(inv_threshold) == float
        train_tab = train_tab[(train_tab.inv > inv_threshold) | (train_tab.label == 0)]

    transform = CorewiseTransform()
    train_ds = BKCorewiseDataset(
        df=train_tab, transform=transform, im_sz=im_sz, style=style
    )
    val_ds = BKCorewiseDataset(df=val_tab, transform=None, im_sz=im_sz, style=style)
    test_ds = BKCorewiseDataset(df=test_tab, transform=None, im_sz=im_sz, style=style)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_sz, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

    return train_dl, val_dl, test_dl

def make_corewise_bk_dataloaders(
    batch_sz,
    im_sz=1024,
    style="avg_all",
    splitting="patients",
    fold=0,
    num_folds=5,
    seed=0,
    inv_threshold=None,
    augmentations=None,
    centers=["UBC", "QUEENS"],
    oversampling=False,
    undersampling=False,
    sampling_ratio=1
):
    if splitting == "patients":
        train_tab, val_tab, test_tab = split_patients(seed=seed)
    elif splitting == "centers":
        train_tab, val_tab, test_tab = split_centerwise()
    elif splitting == "from_file":
        from medAI.datasets.splits.bk_patient_splits import bk_patient_splits as splits
        train_tab = select_patients(splits["train"])
        val_tab = select_patients(splits["val"])
        test_tab = select_patients(splits["test"])
    elif splitting == "from_file_kfold":
        from medAI.datasets.splits.bk_patient_splits import bk_patient_splits as splits
        chosen_split = splits[fold]
        train_tab = select_patients(chosen_split["train"])
        val_tab = select_patients(chosen_split["val"])
        test_tab = select_patients(chosen_split["test"])
    elif splitting == "patients_kfold":
        train_tab, val_tab, test_tab = split_patients_kfold(
            fold, k=num_folds, 
            seed=seed,
            centers=centers,
            oversample_cancer=oversampling, 
            undersample_benign=undersampling, 
            sampling_ratio=sampling_ratio
        )
    else:
        print("Invalid splitting method. Using patients.")
        train_tab, val_tab, test_tab = split_patients(seed=seed)

    if inv_threshold:
        assert type(inv_threshold) == float
        train_tab = train_tab[(train_tab.inv > inv_threshold) | (train_tab.label == 0)]

    train_transform = CorewiseTransform()
    eval_transform = None
    train_ds = BKCorewiseDatasetNew(
        df=train_tab, transform=train_transform, im_sz=im_sz, style=style
    )
    val_ds = BKCorewiseDatasetNew(df=val_tab, transform=eval_transform, im_sz=im_sz, style=style)
    test_ds = BKCorewiseDatasetNew(df=test_tab, transform=eval_transform, im_sz=im_sz, style=style)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_sz, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

    return train_dl, val_dl, test_dl

def make_semi_sl_bk_dataloaders(
    batch_sz,
    im_sz=1024,
    style="first_frame",
    splitting="patients_kfold",
    fold=0,
    num_folds=5,
    seed=0,
    inv_threshold=None,
    centers=["UBC", "QUEENS"],
    oversampling=False,
    undersampling=False,
    sampling_ratio=1
):
    if splitting == "patients":
        train_tab, val_tab, test_tab = split_patients(seed=seed)
    elif splitting == "centers":
        train_tab, val_tab, test_tab = split_centerwise()
    elif splitting == "from_file":
        with open("splits/bk_patient_splits.json", "r") as f:
            splits = json.load(f)
        train_tab = pd.read_csv(splits["train"])
        val_tab = pd.read_csv(splits["val"])
        test_tab = pd.read_csv(splits["test"])
    elif splitting == "from_file_kfold":
        from medAI.datasets.splits.bk_patient_splits import bk_patient_splits as splits
        chosen_split = splits[fold]
        train_tab = select_patients(chosen_split["train"])
        val_tab = select_patients(chosen_split["val"])
        test_tab = select_patients(chosen_split["test"])
    elif splitting == "patients_kfold":
        train_tab, val_tab, test_tab = split_patients_kfold(
            fold, k=num_folds, 
            seed=seed,
            centers=centers,
            oversample_cancer=oversampling, 
            undersample_benign=undersampling, 
            sampling_ratio=sampling_ratio
        )
    else:
        print("Invalid splitting method. Using patients.")
        train_tab, val_tab, test_tab = split_patients(seed=seed)

    if inv_threshold:
        assert type(inv_threshold) == float
        train_tab = train_tab[(train_tab.inv > inv_threshold) | (train_tab.label == 0)]

    train_transform = AugmentThruTime()
    train_ds = BKSemiSupervisedDataset(
        df=train_tab, transform=train_transform, im_sz=im_sz, style=style
    )
    val_ds = BKSemiSupervisedDataset(df=val_tab, transform=None, im_sz=im_sz, style=style)
    test_ds = BKSemiSupervisedDataset(df=test_tab, transform=None, im_sz=im_sz, style=style)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_sz, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

def make_temporal_bk_dataloaders(
    batch_sz,
    im_sz=1024,
    style="first_frame",
    splitting="patients_kfold",
    fold=0,
    num_folds=5,
    seed=0,
    inv_threshold=None,
    centers=["UBC", "QUEENS"],
    oversampling=False,
    undersampling=False,
    sampling_ratio=1
):
    if splitting == "patients":
        train_tab, val_tab, test_tab = split_patients(seed=seed)
    elif splitting == "centers":
        train_tab, val_tab, test_tab = split_centerwise()
    elif splitting == "from_file":
        with open("splits/bk_patient_splits.json", "r") as f:
            splits = json.load(f)
        train_tab = pd.read_csv(splits["train"])
        val_tab = pd.read_csv(splits["val"])
        test_tab = pd.read_csv(splits["test"])
    elif splitting == "from_file_kfold":
        from medAI.datasets.splits.bk_patient_splits import bk_patient_splits as splits
        chosen_split = splits[fold]
        train_tab = select_patients(chosen_split["train"])
        val_tab = select_patients(chosen_split["val"])
        test_tab = select_patients(chosen_split["test"])
    elif splitting == "patients_kfold":
        train_tab, val_tab, test_tab = split_patients_kfold(
            fold, k=num_folds, 
            seed=seed,
            centers=centers,
            oversample_cancer=oversampling, 
            undersample_benign=undersampling, 
            sampling_ratio=sampling_ratio
        )
    else:
        print("Invalid splitting method. Using patients.")
        train_tab, val_tab, test_tab = split_patients(seed=seed)

    if inv_threshold:
        assert type(inv_threshold) == float
        train_tab = train_tab[(train_tab.inv > inv_threshold) | (train_tab.label == 0)]

    train_transform = AugmentThruTime()
    train_ds = BKTemporalDataset(
        df=train_tab, transform=train_transform, im_sz=im_sz, style=style
    )
    val_ds = BKTemporalDataset(df=val_tab, transform=None, im_sz=im_sz, style=style)
    test_ds = BKTemporalDataset(df=test_tab, transform=None, im_sz=im_sz, style=style)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_sz, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

    return train_dl, val_dl, test_dl

class BKCorewiseDataset(Dataset):
    def __init__(
        self,
        df,
        transform,
        im_sz=1024,
        style="avg_all",
        ubc_data=True,
        queens_data=True,
    ):
        super(BKCorewiseDataset, self).__init__()
        self.data = self.collect_files(df)
        self.transform = transform
        self.table = df
        self.im_sz = im_sz
        self.style = style

        self.ubc_data = ubc_data
        self.queens_data = queens_data

    def __getitem__(self, idx):

        file_arr = self.data[idx]
        roi_mask = np.load(file_arr[1])
        prostate_mask = np.load(file_arr[2])
        label = file_arr[3]
        involvement = file_arr[4]
        core_id = file_arr[5]
        patient_id = file_arr[6]
        rf_file = np.load(file_arr[0])

        if self.style == "last_frame":
            bmode = self.make_analytical(rf_file[:, :, -1])
        elif self.style == "avg_last_100":
            bmode = self.make_analytical(rf_file[:, :, -100:].mean(axis=-1))
        elif self.style == "avg_all":
            bmode = self.make_analytical(rf_file.mean(axis=-1))
        elif self.style == "random":
            frame_idx = np.random.randint(100, rf_file.shape[-1])
            bmode = self.make_analytical(rf_file[:, :, frame_idx])
        elif self.style == "random_avg":
            frame_idx = np.random.randint(50, 150)
            bmode = self.make_analytical(
                rf_file[:, :, frame_idx : frame_idx + 5].mean(axis=-1)
            )
        else:
            print("Invalid style. Using avg_all.")
            bmode = self.make_analytical(rf_file.mean(axis=-1))

        bmode = resize(bmode, (self.im_sz, self.im_sz))
        roi_mask = resize(roi_mask, (self.im_sz //4, self.im_sz //4))
        prostate_mask = resize(prostate_mask, (self.im_sz //4, self.im_sz //4))

        if self.transform is not None:
            bmode = torch.from_numpy(bmode).unsqueeze(0).float()
            roi_mask = torch.from_numpy(roi_mask).unsqueeze(0).float()
            prostate_mask = torch.from_numpy(prostate_mask).unsqueeze(0).float()

            bmode, roi_mask, prostate_mask = self.transform(bmode, roi_mask, prostate_mask)
            
            bmode = bmode.squeeze(0).numpy()
            roi_mask = roi_mask.squeeze(0).numpy()
            prostate_mask = prostate_mask.squeeze(0).numpy()

        return (bmode, roi_mask, prostate_mask, label, involvement, core_id, patient_id)

    def __len__(self):
        return len(self.data)

    def collect_files(self, df):
        file_tuples = []
        for filetemplate in list(df.filetemplate):
            try:
                roi_file = filetemplate + "_needle.npy"
                wp_file = filetemplate + "_prostate.npy"
                rf_file = filetemplate + "_rf.npy"
                core_suffix = (
                    filetemplate.split("/")[-1].replace("pat", "").replace("_cor", ".")
                )
                patient_id = int(core_suffix.split(".")[0])
                core_idx = int(core_suffix.split(".")[1])
                center_pre = filetemplate.split("/")[3].split('_')[1].lower()
                core_id = f"{center_pre}_{patient_id:04d}.{core_idx}"
                os.stat(rf_file)
                os.stat(roi_file)
                os.stat(wp_file)
                label_values = df[df.core_id == core_id].label.values
                assert label_values.shape[0] == 1
                sub_df = df[df.core_id == core_id]
                inv = sub_df.inv.values[0]
                core_id = sub_df.core_id.values[0]
                patient_id = sub_df.patient_id.values[0]
                file_tuples.append(
                    (
                        rf_file,
                        roi_file,
                        wp_file,
                        label_values[0],
                        inv,
                        core_id,
                        patient_id,
                    )
                )
            except AssertionError:
                pass
            except FileNotFoundError:
                pass

        print(f"Found {len(file_tuples)} files.")

        return file_tuples

    def collect_files_old(self, df, data_dir, core_idx_upper_bound=15):
        file_tuples = []
        for patient in list(set(df.patient_id)):
            for i in range(1, 12):
                try:
                    roi_file = f"{data_dir}pat{patient}_cor{i}_needle.npy"
                    wp_file = f"{data_dir}pat{patient}_cor{i}_prostate.npy"
                    rf_file = f"{data_dir}pat{patient}_cor{i}_rf.npy"
                    os.stat(rf_file)
                    label_values = df[df.core_id == f"{patient}.{i}"].label.values
                    assert label_values.shape[0] == 1
                    sub_df = df[df.core_id == f"{patient}.{i}"]
                    inv = sub_df.inv.values[0]
                    core_id = sub_df.core_id.values[0]
                    patient_id = sub_df.patient_id.values[0]
                    file_tuples.append(
                        (
                            rf_file,
                            roi_file,
                            wp_file,
                            label_values[0],
                            inv,
                            core_id,
                            patient_id,
                        )
                    )
                except AssertionError:
                    pass
                except FileNotFoundError:
                    pass

        return file_tuples

    def make_analytical(self, x):
        from scipy.signal import hilbert

        return np.abs(hilbert(x)) ** 0.3

class BKCorewiseDatasetNew(BKCorewiseDataset):
    def __getitem__(self, idx):

        file_arr = self.data[idx]
        roi_mask = np.load(file_arr[1])
        prostate_mask = np.load(file_arr[2])
        label = file_arr[3]
        involvement = file_arr[4]
        core_id = file_arr[5]
        patient_id = file_arr[6]
        rf_file = np.load(file_arr[0])

        if self.style == "last_frame":
            bmode = self.make_analytical(rf_file[:, :, -1])
        elif self.style == "avg_last_100":
            bmode = self.make_analytical(rf_file[:, :, -100:].mean(axis=-1))
        elif self.style == "avg_all":
            bmode = self.make_analytical(rf_file.mean(axis=-1))
        elif self.style == "random":
            frame_idx = np.random.randint(100, rf_file.shape[-1])
            bmode = self.make_analytical(rf_file[:, :, frame_idx])
        elif self.style == "random_avg":
            frame_idx = np.random.randint(50, 150)
            bmode = self.make_analytical(
                rf_file[:, :, frame_idx : frame_idx + 5].mean(axis=-1)
            )
        else:
            print("Invalid style. Using avg_all.")
            bmode = self.make_analytical(rf_file.mean(axis=-1))

        bmode = resize(bmode, (self.im_sz, self.im_sz))
        roi_mask = resize(roi_mask, (self.im_sz //4, self.im_sz //4))
        prostate_mask = resize(prostate_mask, (self.im_sz //4, self.im_sz //4))
        ood_mask = ((roi_mask + prostate_mask) < 0.5)

        if self.transform is not None:
            bmode = torch.from_numpy(bmode).unsqueeze(0).float()
            roi_mask = torch.from_numpy(roi_mask).unsqueeze(0).float()
            prostate_mask = torch.from_numpy(prostate_mask).unsqueeze(0).float()
            ood_mask = torch.from_numpy(ood_mask).unsqueeze(0).float()

            bmode, roi_mask, prostate_mask = self.transform(bmode, roi_mask, prostate_mask)
            
            bmode = bmode.squeeze(0).numpy()
            roi_mask = roi_mask.squeeze(0).numpy()
            prostate_mask = prostate_mask.squeeze(0).numpy()
            ood_mask = ood_mask.squeeze(0).numpy()

        return (bmode, roi_mask, prostate_mask, ood_mask, label, involvement, core_id, patient_id)

class BKSemiSupervisedDataset(BKCorewiseDataset):
    def __getitem__(self, idx):

        file_arr = self.data[idx]
        roi_mask = np.load(file_arr[1])
        prostate_mask = np.load(file_arr[2])
        label = file_arr[3]
        involvement = file_arr[4]
        core_id = file_arr[5]
        patient_id = file_arr[6]
        rf_file = np.load(file_arr[0])

        if self.style == "last_frame":
            bmode = self.make_analytical(rf_file[:, :, -1])
        elif self.style == "avg_last_100":
            bmode = self.make_analytical(rf_file[:, :, -100:].mean(axis=-1))
        elif self.style == "avg_all":
            bmode = self.make_analytical(rf_file.mean(axis=-1))
        elif self.style == "random":
            frame_idx = np.random.randint(100, rf_file.shape[-1])
            bmode = self.make_analytical(rf_file[:, :, frame_idx])
        elif self.style == "random_avg":
            frame_idx = np.random.randint(50, 150)
            bmode = self.make_analytical(
                rf_file[:, :, frame_idx : frame_idx + 5].mean(axis=-1)
            )
        else:
            print("Invalid style. Using avg_all.")
            bmode = self.make_analytical(rf_file.mean(axis=-1))

        bmode = resize(bmode, (self.im_sz, self.im_sz))
        roi_mask = resize(roi_mask, (self.im_sz //4, self.im_sz //4))
        prostate_mask = resize(prostate_mask, (self.im_sz //4, self.im_sz //4))
        ood_mask = ((roi_mask + prostate_mask) < 0.5)

        if self.transform is not None:
            bmode = torch.from_numpy(bmode).unsqueeze(0).float()
            roi_mask = torch.from_numpy(roi_mask).unsqueeze(0).float()
            prostate_mask = torch.from_numpy(prostate_mask).unsqueeze(0).float()
            ood_mask = torch.from_numpy(ood_mask).unsqueeze(0).float()

            bmode, roi_mask, prostate_mask = self.transform(bmode, roi_mask, prostate_mask)
            
            bmode = bmode.squeeze(0).numpy()
            roi_mask = roi_mask.squeeze(0).numpy()
            prostate_mask = prostate_mask.squeeze(0).numpy()
            ood_mask = ood_mask.squeeze(0).numpy()

        return (bmode, bmode, roi_mask, prostate_mask, ood_mask, label, involvement, core_id, patient_id)

class BKTemporalDataset(BKCorewiseDataset):
    def __getitem__(self, idx):

        file_arr = self.data[idx]
        roi_mask = np.load(file_arr[1])
        prostate_mask = np.load(file_arr[2])
        label = file_arr[3]
        involvement = file_arr[4]
        core_id = file_arr[5]
        patient_id = file_arr[6]
        rf_file = np.load(file_arr[0])

        if self.style == "first_frame":
            bmode_main = self.make_analytical(rf_file[:, :, 0])
            next_frame = np.random.randint(100, rf_file.shape[-1])
            bmode_next = self.make_analytical(rf_file[:, :, next_frame])
        elif self.style == "first_and_last":
            bmode_main = self.make_analytical(rf_file[:, :, 0])
            bmode_next = self.make_analytical(rf_file[:, :, -1])
        elif self.style == "last_and_all":
            bmode_main = self.make_analytical(rf_file[:, :, -1])
            bmode_next = self.make_analytical(rf_file[:, :, :-1].mean(axis=-1))
        elif self.style == "first_and_all":
            bmode_main = self.make_analytical(rf_file[:, :, 0])
            bmode_next = self.make_analytical(rf_file[:, :, :].mean(axis=-1))
        elif self.style == "first_and_random":
            bmode_main = self.make_analytical(rf_file[:, :, 0])
            next_frame = np.random.randint(100, rf_file.shape[-1])
            bmode_next = self.make_analytical(rf_file[:, :, next_frame])
        elif self.style == "last_and_random":
            bmode_main = self.make_analytical(rf_file[:, :, -1])
            next_frame = np.random.randint(100, rf_file.shape[-1])
            bmode_next = self.make_analytical(rf_file[:, :, next_frame])
        elif self.style == "random":
            frame_idx = np.random.randint(0, rf_file.shape[-1])
            bmode_main = self.make_analytical(rf_file[:, :, frame_idx])
            next_frame = np.random.randint(frame_idx, rf_file.shape[-1])
            bmode_next = self.make_analytical(rf_file[:, :, next_frame])
        else:
            print("Invalid style. Using first_frame.")
            bmode_main = self.make_analytical(rf_file[:, :, 0])
            next_frame = np.random.randint(100, rf_file.shape[-1])
            bmode_next = self.make_analytical(rf_file[:, :, next_frame])

        bmode1 = resize(bmode_main, (self.im_sz, self.im_sz))
        bmode2 = resize(bmode_next, (self.im_sz, self.im_sz))
        roi_mask = resize(roi_mask, (self.im_sz //4, self.im_sz //4))
        prostate_mask = resize(prostate_mask, (self.im_sz //4, self.im_sz //4))
        ood_mask = ((roi_mask + prostate_mask) < 0.5)

        if self.transform is not None:
            bmode1 = torch.from_numpy(bmode1).unsqueeze(0).float()
            bmode2 = torch.from_numpy(bmode2).unsqueeze(0).float()
            roi_mask = torch.from_numpy(roi_mask).unsqueeze(0).float()
            prostate_mask = torch.from_numpy(prostate_mask).unsqueeze(0).float()
            ood_mask = torch.from_numpy(ood_mask).unsqueeze(0).float()

            random_translation = RandomTranslation()
            bmode1 = self.transform("weak", bmode1)
            bmode2 = self.transform("strong", bmode2)
            bmode1, bmode2, roi_mask, prostate_mask, ood_mask = random_translation(bmode1, bmode2, roi_mask, prostate_mask, ood_mask)
            
            bmode1 = bmode1.squeeze(0).numpy()
            bmode2 = bmode2.squeeze(0).numpy()

            roi_mask = roi_mask.squeeze(0).numpy()
            prostate_mask = prostate_mask.squeeze(0).numpy()
            ood_mask = ood_mask.squeeze(0).numpy()

        return (bmode1, bmode2, roi_mask, prostate_mask, ood_mask, label, involvement, core_id, patient_id)

class BKPatchDataset(Dataset):
    def __init__(
        self,
        data_dir,
        patient_ids,
        transform,
        pid_range=(0, np.Inf),
        norm=True,
        return_idx=True,
        stats=None,
        slide_idx=-1,
        time_series=False,
        pid_excluded=None,
        return_prob=False,
        tta=False,
        *args,
        **kwargs,
    ):
        super(BKPatchDataset, self).__init__()
        # self.files = glob(f'{data_dir}/*/*/*/*.npy')
        data_dir = data_dir.replace("\\", "/")
        self.files = select_patients(
            glob(f"{data_dir}/*/patches_rf_core/*/*.npy"), patient_ids
        )
        self.transform = transform
        self.pid_range = pid_range
        self.pid_excluded = pid_excluded
        self.norm = norm
        self.pid, self.cid, self.inv, self.label = [], [], [], None
        self.attrs = ["files", "pid", "cid"]
        self.stats = stats
        self.slide_idx = slide_idx
        self.time_series = time_series
        self.return_idx = return_idx
        self.return_prob = return_prob
        self.probability = None
        self.tta = tta  # test time augmentation

    def extract_metadata(self):
        for file in self.files:
            self.pid.append(int(re.findall("/Patient(\d+)/", file)[0]))
            self.cid.append(int(re.findall("/core(\d+)", file)[0]))
            self.inv.append(float(re.findall("_inv([\d.[0-9]+)", file)[0]))
        for attr in self.attrs:  # convert to array
            setattr(self, attr, np.array(getattr(self, attr)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], mmap_mode="c").astype("float32")

        if self.transform is not None:
            data = self.transform(data)
        if self.time_series:
            data = (
                F.avg_pool2d(torch.tensor(data), kernel_size=(8, 8), stride=8)
                .flatten(1)
                .T
            )
            if self.norm:
                data = (data - np.median(data)) / (
                    np.percentile(data, 75) - np.percentile(data, 25)
                )
        if self.norm and not self.time_series:
            if isinstance(data, tuple) or isinstance(data, list):
                data = tuple(self.norm_data(d) for d in data)
            else:
                data = self.norm_data(data)
            data = (data - np.median(data)) / (
                np.percentile(data, 75) - np.percentile(data, 25)
            )

        if self.tta:
            data = np.concatenate([data, np.flip(data, axis=-1)], axis=0)

        if self.label is not None:
            label = self.label[idx]
            if self.return_prob:
                assert self.probability is not None
                return data, label, self.probability[idx]
            if self.return_idx:
                return data, label, idx, self.pid[idx], self.cid[idx], self.inv[idx]
            return data, label

        return data[0], data[1]

    def norm_data(self, data):
        if self.stats is not None:
            data = (data - self.stats[0]) / self.stats[1]
        else:
            data = (data - data.mean()) / data.std()
        return data

    def filter_by_pid(self):
        idx = np.logical_and(
            self.pid >= self.pid_range[0], self.pid <= self.pid_range[1]
        )
        if self.pid_excluded is not None:
            idx[np.isin(self.pid, self.pid_excluded)] = False
        self.filter_by_idx(idx)

    def filter_by_idx(self, idx):
        for attr in self.attrs:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr)[idx])


class BKPatchLabeledDataset(BKPatchDataset):
    def __init__(
        self,
        data_dir,
        patient_ids,
        transform=None,
        pid_range=(0, np.Inf),
        inv_range=(0, 1),
        gs_range=(7, 10),
        queens_data=False,
        file_idx=None,
        oversampling_cancer=False,
        *args,
        **kwargs,
    ):
        super().__init__(data_dir, patient_ids, transform, pid_range, *args, **kwargs)
        self.inv_range = inv_range
        self.gs_range = gs_range
        self.attrs.extend(["label", "gs", "location", "id", "inv"])
        self.label, self.inv, self.gs, self.location, self.id = [[] for _ in range(5)]
        self.queens_data = queens_data

        oversampling_dict = {0.8: 22, 0.7: 17, 0.6: 11, 0.5: 7, 0.4: 6}
        if oversampling_cancer:
            oversampling_rate = oversampling_dict[min(self.inv_range)]
            # the oversampling rate (17) is calculated based on the class ratio after all filtering steps
            oversampled_files = []
            for file in self.files:
                if "_cancer" in file:
                    oversampled_files.extend([file for _ in range(oversampling_rate)])
            self.files += oversampled_files

        self.extract_metadata()
        self.filter_by_pid()
        self.filter_by_inv()
        self.filter_by_gs()
        if file_idx is not None:
            self.filter_by_idx(file_idx)

    def extract_metadata(self):
        for file in self.files:
            folder_name = os.path.basename(os.path.dirname(file))
            (
                self.label.append(0)
                if folder_name.split("_")[1] == "benign"
                else self.label.append(1)
            )
            self.location.append(folder_name.split("_")[-2])
            self.inv.append(float(re.findall("_inv([\d.[0-9]+)", file)[0]))
            self.gs.append(int(re.findall("_gs(\d+)", file)[0]))
            self.pid.append(int(re.findall("/Patient(\d+)/", file)[0]))
            self.cid.append(int(re.findall("/core(\d+)_", file)[0]))
            self.id.append(int(folder_name.split("_")[-1][2:]))
        for attr in self.attrs:  # convert to array
            setattr(self, attr, np.array(getattr(self, attr)))

    def filter_by_inv(self):
        idx = np.logical_and(
            self.inv >= self.inv_range[0], self.inv <= self.inv_range[1]
        )
        idx = np.logical_or(idx, self.inv == 0)
        self.filter_by_idx(idx)

    def filter_by_gs(self):
        idx = np.logical_and(self.gs >= self.gs_range[0], self.gs <= self.gs_range[1])
        idx = np.logical_or(idx, self.gs == 0)
        self.filter_by_idx(idx)


class BKPatchUnlabeledDataset(BKPatchDataset):
    def __init__(
        self,
        data_dir,
        patient_ids,
        transform=None,
        pid_range=(0, np.Inf),
        stats=None,
        norm=True,
        *args,
        **kwargs,
    ):
        super(BKPatchUnlabeledDataset, self).__init__(
            data_dir, patient_ids, transform, pid_range, norm=norm
        )
        # Note: cid: per patient core id; id: absolute core id
        self.extract_metadata()
        self.filter_by_pid()


class PatchSSLTransform:
    def __call__(self, patch):
        patch = torch.from_numpy(patch).float() / 255.0
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)

        augs = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        ]
        p1 = T.Compose(augs)(patch)
        p2 = T.Compose(augs)(patch)

        return p1, p2


class PatchTransform:
    def __call__(self, patch):
        patch = torch.from_numpy(patch).float() / 255.0
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)
        return patch


class Identity:
    def __call__(self, *images):
        return images[0] if len(images) == 1 else images


class SpeckleNoise:
    def __call__(self, *images):
        from torchvision.transforms.functional import affine
        from random import uniform

        outputs = []
        for image in images:
            C, H, W = image.shape
            gauss = torch.randn((C, H, W))
            noisy = image + 0.5 * image * gauss
            outputs.append(noisy)

        return outputs[0] if len(outputs) == 1 else outputs


class RandomTranslation:
    def __init__(self, translation=(0.1, 0.1)):
        self.translation = translation

    def __call__(self, *images):
        from torchvision.transforms.functional import affine
        from random import uniform

        h_factor, w_factor = uniform(
            -self.translation[0], self.translation[0]
        ), uniform(-self.translation[1], self.translation[1])

        outputs = []
        for image in images:
            H, W = image.shape[-2:]
            translate_x = int(w_factor * W)
            translate_y = int(h_factor * H)
            outputs.append(
                affine(
                    image,
                    angle=0,
                    translate=(translate_x, translate_y),
                    scale=1,
                    shear=0,
                )
            )

        return outputs[0] if len(outputs) == 1 else outputs


class CorewiseTransform:
    def __call__(self, *images):
        transform = RandomTranslation()

        return transform(*images)

class AugmentThruTime:
    def __call__(self, mode, *images):
        if mode == "weak":
            weak = WeakAugment()
            return weak(*images)
        elif mode == "strong":
            strong = StrongAugment()
            return strong(*images)
        
        return images


class WeakAugment:
    def __init__(self):
        augs = [
            RandomTranslation(),
        ]
        
        self.aug = T.RandomChoice(augs)
    
    def __call__(self, *images):
        outputs = []
        for image in images:
            H, W = image.shape[-2:]
            image = self.aug(image)
            outputs.append(image)

        return outputs[0] if len(outputs) == 1 else outputs 

class StrongAugment:
    def __init__(self):
        def h_linecut(img, p=0.2, size=128):
            im=img.clone()
            if torch.rand(1) > p:
                return im
            x = torch.randint(0, im.shape[1] - size, (1,))
            im[:, x:x+size, :] = 0
            return im

        def v_linecut(img, p=0.2, size=128):
            im=img.clone()
            if torch.rand(1) > p:
                return im
            y = torch.randint(0, im.shape[2] - size, (1,))
            im[:, :, y:y+size] = 0
            return im

        def pixel_cut(img, p=0.4, pixelp=0.03):
            im=img.clone()
            if torch.rand(1) > p:
                return im
            prob = torch.rand_like(im)
            mask = prob < pixelp
            im[mask] = 0
            return im
        
        def pixel_saturation(img, p=0.4, pixelp=0.03):
            im=img.clone()
            if torch.rand(1) > p:
                return im
            prob = torch.rand_like(im)
            mask = prob < pixelp
            im[mask] = im.max()
            return im

        self.aug = T.Compose([
            T.ColorJitter(hue=0.3),
            T.Lambda(lambda x: pixel_cut(x)),
            T.Lambda(lambda x: pixel_saturation(x)),
            T.Lambda(lambda x: h_linecut(x, size=2)),
            T.Lambda(lambda x: h_linecut(x, size=2)),
            T.Lambda(lambda x: h_linecut(x, size=2)),
            T.Lambda(lambda x: h_linecut(x, size=2)),
            T.Lambda(lambda x: v_linecut(x, size=2)),
            T.Lambda(lambda x: v_linecut(x, size=2)),
            T.Lambda(lambda x: v_linecut(x, size=2)),
            T.Lambda(lambda x: v_linecut(x, size=2)),
        ])
    
    def __call__(self, *images):
        outputs = []
        for image in images:
            H, W = image.shape[-2:]
            outputs.append(self.aug(image))

        return outputs[0] if len(outputs) == 1 else outputs