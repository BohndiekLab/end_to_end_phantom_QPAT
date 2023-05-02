import torch
import glob
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage.morphology import distance_transform_edt


class PalpaitineDataset(Dataset):

    # Randomised but reproducible fold-partitions for the 84 phantoms in the training data set
    fold = {
        0: [ 2, 79,  5, 66, 55, 45, 62, 26, 18, 75, 73, 24, 39, 36, 48, 33],
        1: [37, 67, 13, 71,  3,  1, 69, 78, 54, 72, 11, 25, 34, 40, 12, 51],
        2: [19, 30, 83, 57, 74, 53, 41, 82, 20, 31, 28, 76, 81, 64, 42, 52],
        3: [65, 43,  6, 68, 15,  8,  4, 17, 44, 14, 27, 23, 80, 56,  0, 49],
        4: [38, 63, 32, 60, 29, 35,  9, 21, 22, 47, 10, 77, 61, 50,  7, 59]
    }

    def __init__(self,
                 data_path,
                 fold=0,
                 train=True,
                 device="cpu",
                 augment=False,
                 use_all_data=False,
                 mean_musp=0,
                 std_musp=1,
                 mean_mua=0,
                 std_mua=1,
                 mean_signal=0,
                 std_signal=1,
                 mean_fluence=0,
                 std_fluence=1,
                 experimental_data=False):
        self.device = device
        files = glob.glob(data_path + "/*.npz")
        files.sort()
        if not use_all_data:
            tmp_files = []
            if train:
                for idx in range(int(len(files)/21)):
                    if not idx in self.fold[fold]:
                        tmp_files += files[idx*21:(idx+1)*21]
            else:
                for idx in range(int(len(files) / 21)):
                    if idx in self.fold[fold]:
                        tmp_files += files[idx * 21:(idx + 1) * 21]
            files = tmp_files

        print(f"Found {len(files)} items. Loading data...")
        images = [None] * len(files)
        segmentations = [None] * len(files)
        absorptions = [None] * len(files)
        scatterings = [None] * len(files)
        fluences = [None] * len(files)
        for file_idx, file in enumerate(files):
            # if file_idx % 21 == 0:
            #   print(file)
            print("\r", file_idx+1, "/", len(files), end='', flush=True)
            np_data = np.load(file)
            fluence = np_data["fluence"]
            if experimental_data:
                images[file_idx] = (np_data["features_das"] - mean_signal) / std_signal
                segmentations[file_idx] = np_data["segmentation"]
                absorptions[file_idx] = (np_data["mua"] - mean_mua) / std_mua
                scatterings[file_idx] = (np_data["musp"] - mean_musp) / std_musp
                fluences[file_idx] = (fluence - mean_fluence) / std_fluence
            else:
                images[file_idx] = (np_data["features_sim"] - mean_signal) / std_signal
                segmentations[file_idx] = np_data["segmentation"]
                absorptions[file_idx] = (np_data["mua"] - mean_mua) / std_mua
                scatterings[file_idx] = (np_data["musp"] - mean_musp) / std_musp
                fluences[file_idx] = (fluence - mean_fluence) / std_fluence


        if train and augment:
            images = images + [np.fliplr(image) for image in images]
            segmentations = segmentations + [np.fliplr(image) for image in segmentations]
            absorptions = absorptions + [np.fliplr(image) for image in absorptions]
            scatterings = scatterings + [np.fliplr(image) for image in scatterings]
            fluences = fluences + [np.fliplr(image) for image in fluences]

        print("")

        self.images = torch.from_numpy(np.asarray(images).reshape(-1, 1, 288, 288)).float()
        segmentations = np.asarray(segmentations).astype(int)
        if np.min(segmentations) == -1:
            segmentations = segmentations + 1
        self.instance_segmentations = np.copy(segmentations).reshape(-1, 1, 288, 288)
        segmentations[segmentations > 2] = 2
        num_classes = int(np.round(np.max(segmentations) + 1))
        self.segmentations = torch.from_numpy(segmentations).long()
        self.segmentations = self.segmentations.reshape(-1, 1, 288, 288)
        self.segmentations = torch.nn.functional.one_hot(self.segmentations)
        self.segmentations = torch.permute(self.segmentations, [0, 4, 2, 3, 1])
        self.segmentations = self.segmentations.reshape(-1, num_classes, 288, 288).float()
        self.absorptions = torch.from_numpy(np.asarray(absorptions).reshape(-1, 1, 288, 288)).float()
        self.scatterings = torch.from_numpy(np.asarray(scatterings).reshape(-1, 1, 288, 288)).float()
        self.fluences = torch.from_numpy(np.asarray(fluences).reshape(-1, 1, 288, 288)).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (self.images[idx].to(self.device),
                self.segmentations[idx].to(self.device),
                self.absorptions[idx].to(self.device),
                self.scatterings[idx].to(self.device),
                self.fluences[idx].to(self.device),
                self.instance_segmentations[idx])


class PalpaitineMouseDataset(Dataset):

    def __init__(self,
                 data_path,
                 device="cpu",
                 mean_signal=0,
                 std_signal=1):
        self.device = device
        files = glob.glob(data_path + "/*.npz")
        files.sort()
        print(f"Found {len(files)} items. Loading data...")
        images = [None] * len(files)
        for file_idx, file in enumerate(files):
            print("\r", file_idx+1, "/", len(files), end='', flush=True)
            np_data = np.load(file)["data"]
            images[file_idx] = (np_data - mean_signal) / std_signal

        print("")

        self.images = torch.from_numpy(np.asarray(images).reshape(-1, 1, 288, 288)).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx].to(self.device)


def load_all_sim_and_exp_signals(path):
    exp_signals = []
    sim_signals = []
    labels = []

    files = glob.glob(path + "*_800.npz")

    for path in files:
        data = np.load(path)
        seg = data["segmentation"]
        img = data["features_das"]
        sim_data = np.load(path.replace("final_data", "final_data/sim"))
        sim_img = sim_data["features"]
        sim_img = np.fliplr(sim_img.T)

        label = path.split("\\")[-1].split("/")[-1].split("_800.npz")[0]

        if len(img[seg > 1]) > 0:
            for x in np.unique(seg[seg > 1]):
                distance_mapping = distance_transform_edt(seg==x)
                exp_signals.append(np.percentile(img[seg == x], 50))
                sim_signals.append(np.percentile(sim_img[(seg == x) & (distance_mapping < 6)], 50))
                labels.append(label + "_" + str(x))

    return np.asarray(exp_signals), np.asarray(sim_signals), labels
