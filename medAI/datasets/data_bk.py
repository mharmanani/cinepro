import re
import os
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

DATA_DIR_ROOT_MAIN = "/ssd005/projects/exactvu_pca/bk_ubc/"
DATA_DIR_PATCH_ROOT =  "/ssd005/projects/exactvu_pca/bk_ubc/patches/UBC/patch_48x48_str32_avg/"

def get_patch_labels():
    patients = {}
    for patient in os.listdir(DATA_DIR_PATCH_ROOT):
        labels = []
        invs = []
        try:
            for core in os.listdir(DATA_DIR_PATCH_ROOT + str(patient) + "/patches_rf_core"):
                labels.append(1 if "cancer" in core else 0)
                inv = re.findall('_inv([\d.[0-9]+)', core)
                invs.append(float(inv[0]) if inv else 0)
            patients[patient] = (labels, invs)
        except FileNotFoundError:
            continue

    return patients

def build_label_table():
    df = pd.DataFrame()
    for key, value in get_patch_labels().items():
        labels, invs = value
        for i in range(len(labels)):
            label, inv = labels[i], invs[i]
            row = pd.DataFrame({"core_id": [key+str(i)], "patient_id": [key], "label": [label], "inv": [inv]})
            df = pd.concat([df, row], ignore_index=True)
    return df

def select_patients(all_files, patient_ids):
    patient_files = []
    for file in all_files:
        patient = int(re.findall('/Patient(\d+)/', file)[0])
        if patient in patient_ids:
            patient_files.append(file)
    return patient_files

def split_patients(inv_threshold=None):
    table = build_label_table()
    if inv_threshold:
        table = table[(table["inv"] >= inv_threshold) | (table["label"] == 0)]
    patient_table = table.drop_duplicates(subset=["patient_id"])

    train_pa, val_pa = train_test_split(patient_table, 
        test_size=0.3, random_state=0, 
        stratify=patient_table["label"])
    
    val_pa, test_pa = train_test_split(val_pa, 
        test_size=0.5, random_state=0, 
        stratify=val_pa["label"])

    train_idx = table.patient_id.isin(train_pa.patient_id)
    val_idx = table.patient_id.isin(val_pa.patient_id)
    test_idx = table.patient_id.isin(test_pa.patient_id)
    
    train_tab, val_tab, test_tab = table[train_idx], table[val_idx], table[test_idx]

    assert set(train_tab.patient_id) & set(val_tab.patient_id) == set()
    assert set(train_tab.patient_id) & set(test_tab.patient_id) == set()
    assert set(val_tab.patient_id) & set(test_tab.patient_id) == set()

    return train_tab, val_tab, test_tab


def get_tmi23_patients(inv_threshold=None):
    def _format_patient_list(lst):
        return [f"Patient{pid}" for pid in lst]
    table = build_label_table()
    if inv_threshold:
        table = table[(table["inv"] >= inv_threshold) | (table["label"] == 0)]

    train_pa = _format_patient_list([2, 3, 5, 6, 8, 11, 13, 14, 15, 16, 18, 20, 22, 23, 24, 26, 27, 28, 38, 39, 40, 41, 42, 43, 44, 45, 48, 50, 51, 52, 53, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 68, 69, 70, 71, 72, 74, 76, 78, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 100])
    val_pa = _format_patient_list([130, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 117, 121, 123, 124, 125, 126, 127])
    test_pa = _format_patient_list([131, 132, 133, 134, 136, 138, 139, 141, 142, 143, 144, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 171, 172, 173, 175, 176, 181, 182, 186, 187])

    train_idx = table.patient_id.isin(set(train_pa))
    val_idx = table.patient_id.isin(val_pa)
    test_idx = table.patient_id.isin(test_pa)

    train_tab, val_tab, test_tab = table[train_idx], table[val_idx], table[test_idx]

    assert set(train_tab.patient_id) & set(val_tab.patient_id) == set()
    assert set(train_tab.patient_id) & set(test_tab.patient_id) == set()
    assert set(val_tab.patient_id) & set(test_tab.patient_id) == set()

    return train_tab, val_tab, test_tab

def make_bk_dataloaders(self_supervised=False, inv_threshold=None):
    train_tab, val_tab, test_tab = get_tmi23_patients(inv_threshold)

    _BKPatchesDataset = BKPatchLabeledDataset if not self_supervised else BKPatchUnlabeledDataset
    transform =PatchTransform() if not self_supervised else PatchSSLTransform()
    train_ds = _BKPatchesDataset(DATA_DIR_PATCH_ROOT, patient_ids=train_tab.patient_id, transform=transform)
    val_ds = _BKPatchesDataset(DATA_DIR_PATCH_ROOT, patient_ids=val_tab.patient_id, transform=transform)
    test_ds = _BKPatchesDataset(DATA_DIR_PATCH_ROOT, patient_ids=test_tab.patient_id, transform=transform)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_dl, val_dl, test_dl

class BKPatchDataset(Dataset):
    def __init__(self, data_dir, patient_ids, transform, pid_range=(0, np.Inf), norm=True, return_idx=True, stats=None,
                 slide_idx=-1, time_series=False, pid_excluded=None, return_prob=False,
                 tta=False, *args, **kwargs):
        super(BKPatchDataset, self).__init__()
        # self.files = glob(f'{data_dir}/*/*/*/*.npy')
        data_dir = data_dir.replace('\\', '/')
        self.files = select_patients(glob(f'{data_dir}/*/patches_rf_core/*/*.npy'), patient_ids)
        self.transform = transform
        self.pid_range = pid_range
        self.pid_excluded = pid_excluded
        self.norm = norm
        self.pid, self.cid, self.inv, self.label = [], [], [], None
        self.attrs = ['files', 'pid', 'cid']
        self.stats = stats
        self.slide_idx = slide_idx
        self.time_series = time_series
        self.return_idx = return_idx
        self.return_prob = return_prob
        self.probability = None
        self.tta = tta  # test time augmentation

    def extract_metadata(self):
        for file in self.files:
            self.pid.append(int(re.findall('/Patient(\d+)/', file)[0]))
            self.cid.append(int(re.findall('/core(\d+)', file)[0]))
            self.inv.append(float(re.findall('_inv([\d.[0-9]+)', file)[0]))
        for attr in self.attrs:  # convert to array
            setattr(self, attr, np.array(getattr(self, attr)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], mmap_mode='c').astype('float32')

        if self.transform is not None:
            data = self.transform(data)
        if self.time_series:
            data = F.avg_pool2d(torch.tensor(data), kernel_size=(8, 8), stride=8).flatten(1).T
            if self.norm:
                data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))
        if self.norm and not self.time_series:
            if isinstance(data, tuple) or isinstance(data, list):
                data = tuple(self.norm_data(d) for d in data)
            else:
                data = self.norm_data(data)
            data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))

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
        idx = np.logical_and(self.pid >= self.pid_range[0], self.pid <= self.pid_range[1])
        if self.pid_excluded is not None:
            idx[np.isin(self.pid, self.pid_excluded)] = False
        self.filter_by_idx(idx)

    def filter_by_idx(self, idx):
        for attr in self.attrs:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr)[idx])

class BKPatchLabeledDataset(BKPatchDataset):
    def __init__(self, data_dir, patient_ids, transform=None, pid_range=(0, np.Inf), inv_range=(0, 1), gs_range=(7, 10),
                 queens_data=False, file_idx=None, oversampling_cancer=False, *args, **kwargs):
        super().__init__(data_dir, patient_ids, transform, pid_range, *args, **kwargs)
        self.inv_range = inv_range
        self.gs_range = gs_range
        self.attrs.extend(['label', 'gs', 'location', 'id', 'inv'])
        self.label, self.inv, self.gs, self.location, self.id = [[] for _ in range(5)]
        self.queens_data = queens_data

        oversampling_dict = {0.8: 22, 0.7: 17, 0.6: 11, 0.5: 7, 0.4: 6}
        if oversampling_cancer:
            oversampling_rate = oversampling_dict[min(self.inv_range)]
            # the oversampling rate (17) is calculated based on the class ratio after all filtering steps
            oversampled_files = []
            for file in self.files:
                if '_cancer' in file:
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
            self.label.append(0) if folder_name.split('_')[1] == 'benign' else self.label.append(1)
            self.location.append(folder_name.split('_')[-2])
            self.inv.append(float(re.findall('_inv([\d.[0-9]+)', file)[0]))
            self.gs.append(int(re.findall('_gs(\d+)', file)[0]))
            self.pid.append(int(re.findall('/Patient(\d+)/', file)[0]))
            self.cid.append(int(re.findall('/core(\d+)_', file)[0]))
            self.id.append(int(folder_name.split('_')[-1][2:]))
        for attr in self.attrs:  # convert to array
            setattr(self, attr, np.array(getattr(self, attr)))

    def filter_by_inv(self):
        idx = np.logical_and(self.inv >= self.inv_range[0], self.inv <= self.inv_range[1])
        idx = np.logical_or(idx, self.inv == 0)
        self.filter_by_idx(idx)

    def filter_by_gs(self):
        idx = np.logical_and(self.gs >= self.gs_range[0], self.gs <= self.gs_range[1])
        idx = np.logical_or(idx, self.gs == 0)
        self.filter_by_idx(idx)

class BKPatchUnlabeledDataset(BKPatchDataset):
    def __init__(self, data_dir, patient_ids, transform=None, pid_range=(0, np.Inf), stats=None, norm=True, *args, **kwargs):
        super(BKPatchUnlabeledDataset, self).__init__(data_dir, patient_ids, transform, pid_range, norm=norm)
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