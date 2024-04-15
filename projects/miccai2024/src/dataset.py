from medAI.datasets.nct2013.utils import load_or_create_resized_bmode_data
from torch.utils.data import Dataset
from torchvision import transforms as T
from medAI.datasets.nct2013 import data_accessor
import numpy as np 
from medAI.utils.data.patch_extraction import PatchView
import torch

import glob


class BKPatchDataset(Dataset):
    def __init__(self, data_dir, transform, pid_range=(0, np.Inf), norm=True, return_idx=True, stats=None,
                 slide_idx=-1, time_series=False, pid_excluded=None, return_prob=False,
                 tta=False, *args, **kwargs):
        super(BKPatchDataset, self).__init__()
        # self.files = glob(f'{data_dir}/*/*/*/*.npy')
        data_dir = data_dir.replace('\\', '/')
        self.files = glob(f'{data_dir}/*/patches_rf_core/*/*.npy')
        self.transform = transform
        self.pid_range = pid_range
        self.pid_excluded = pid_excluded
        self.norm = norm
        self.pid, self.cid, self.label = [], [], None
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
                return data, label, idx
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
    def __init__(self, data_dir, transform=None, pid_range=(0, np.Inf), inv_range=(0, 1), gs_range=(7, 10),
                 queens_data=False, file_idx=None, oversampling_cancer=False, *args, **kwargs):
        super().__init__(data_dir, transform, pid_range, *args, **kwargs)
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
    def __init__(self, data_dir, transform=None, pid_range=(0, np.Inf), stats=None, norm=True, *args, **kwargs):
        super(BKPatchUnlabeledDataset, self).__init__(data_dir, transform, pid_range, norm=norm)
        # Note: cid: per patient core id; id: absolute core id
        self.extract_metadata()
        self.filter_by_pid()


class CropFixSize:
    def __init__(self, sz=32, in_channels=1):
        import imgaug.augmenters as iaa
        self.in_channels = in_channels
        self.seq = iaa.CenterCropToFixedSize(sz, sz)

    def __call__(self, sample):

        if (sample.ndim > 2) and (sample.shape[-1] > 1):
            assert sample.shape[-1] >= self.in_channels
            sample = sample[..., :self.in_channels]

        x1 = self.seq(image=sample)

        if x1.ndim > 2:
            return x1.transpose([2, 0, 1])
        return x1[np.newaxis]

class BModePatchesDataset(Dataset):
    _bmode_data, _core_id_2_idx = load_or_create_resized_bmode_data((1024, 1024))
    _metadata_table = data_accessor.get_metadata_table()

    def __init__(
        self,
        core_ids,
        patch_size,
        stride,
        needle_mask_threshold,
        prostate_mask_threshold,
        transform=None,
    ):
        self.core_ids = sorted(core_ids)
        N = len(self.core_ids)

        self._images = [
            self._bmode_data[self._core_id_2_idx[core_id]] for core_id in core_ids
        ]
        self._prostate_masks = np.zeros((N, 256, 256))
        for i, core_id in enumerate(core_ids):
            self._prostate_masks[i] = data_accessor.get_prostate_mask(core_id)
        self._needle_masks = np.zeros((N, 512, 512))
        for i, core_id in enumerate(core_ids):
            self._needle_masks[i] = data_accessor.get_needle_mask(core_id)
        self._patch_views = PatchView.build_collection_from_images_and_masks(
            self._images,
            window_size=patch_size,
            stride=stride,
            align_to="topright",
            mask_lists=[self._prostate_masks, self._needle_masks],
            thresholds=[prostate_mask_threshold, needle_mask_threshold],
        )

        self._metadata_dicts = []
        for core_id in self.core_ids:
            metadata = (
                self._metadata_table[self._metadata_table.core_id == core_id]
                .iloc[0]
                .to_dict()
            )
            self._metadata_dicts.append(metadata)

        self._indices = []
        for i, pv in enumerate(self._patch_views):
            self._indices.extend([(i, j) for j in range(len(pv))])

        self.transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, j = self._indices[idx]
        pv = self._patch_views[i]

        item = {}
        item["patch"] = pv[j] / 255.0

        metadata = self._metadata_dicts[i].copy()
        item.update(metadata)

        if self.transform is not None:
            item = self.transform(item)
        return item


class RFPatchesDataset(Dataset):
    _metadata_table = data_accessor.get_metadata_table()

    def __init__(
        self,
        core_ids,
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        needle_mask_threshold=0.6,
        prostate_mask_threshold=-1,
        transform=None,
    ):
        self.core_ids = core_ids
        im_size_mm = 28, 46.06
        im_size_px = data_accessor.get_rf_image(core_ids[0], 0).shape
        self.patch_size_px = int(patch_size_mm[0] * im_size_px[0] / im_size_mm[0]), int(
            patch_size_mm[1] * im_size_px[1] / im_size_mm[1]
        )
        self.patch_stride_px = int(
            patch_stride_mm[0] * im_size_px[0] / im_size_mm[0]
        ), int(patch_stride_mm[1] * im_size_px[1] / im_size_mm[1])

        self._images = [data_accessor.get_rf_image(core_id, 0) for core_id in core_ids]
        self._prostate_masks = [
            data_accessor.get_prostate_mask(core_id) for core_id in core_ids
        ]
        self._needle_masks = [
            data_accessor.get_needle_mask(core_id) for core_id in core_ids
        ]

        self._patch_views = PatchView.build_collection_from_images_and_masks(
            self._images,
            window_size=self.patch_size_px,
            stride=self.patch_stride_px,
            align_to="topright",
            mask_lists=[self._prostate_masks, self._needle_masks],
            thresholds=[prostate_mask_threshold, needle_mask_threshold],
        )
        self._indices = []
        for i, pv in enumerate(self._patch_views):
            self._indices.extend([(i, j) for j in range(len(pv))])

        self.transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, j = self._indices[idx]
        metadata = (
            self._metadata_table[self._metadata_table.core_id == self.core_ids[i]]
            .iloc[0]
            .to_dict()
        )
        pv = self._patch_views[i]
        patch = pv[j]

        patch = patch.copy()
        from skimage.transform import resize
        resize(patch, (256, 256))    
        postition = pv.positions[j]

        data = {"patch": patch, **metadata, "position": postition}
        if self.transform is not None:
            data = self.transform(data)

        return data


class SSLTransform:
    def __call__(self, item):
        patch = item["patch"]
        patch = torch.from_numpy(patch).float()
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)

        augs = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        ]
        p1 = T.Compose(augs)(patch)
        p2 = T.Compose(augs)(patch)

        return p1, p2


class Transform:
    def __call__(self, item):
        patch = item["patch"]
        patch = torch.from_numpy(patch).float() 
        patch = patch.unsqueeze(0).repeat_interleave(3, dim=0)
        item["patch"] = patch
        return item

