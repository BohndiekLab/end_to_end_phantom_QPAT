import torch
from torch.utils.data import DataLoader
from utils.data_loading import PalpaitineDataset, PalpaitineMouseDataset
from utils.networks import RegressionUNet
import numpy as np
import os

EXPERIMENTAL_DATA = False
REEVALUATE_TEST_DATA = False
REEVALUATE_FLOW_DATA = True
REEVALUATE_MOUSE_DATA = False

if EXPERIMENTAL_DATA:
    BASE_PATH = "experimental/fold_"
else:
    BASE_PATH = "simulation/fold_"

train_data = PalpaitineDataset(data_path=f"C:/final_data_simulation/training",
                               augment=False, use_all_data=True, experimental_data=True)

mean_musp = (np.mean(train_data.scatterings.detach().cpu().numpy()))
std_musp = (np.std(train_data.scatterings.detach().cpu().numpy()))
mean_mua = (np.mean(train_data.absorptions.detach().cpu().numpy()))
std_mua = (np.std(train_data.absorptions.detach().cpu().numpy()))
mean_signal = (np.mean(train_data.images.detach().cpu().numpy()))
std_signal = (np.std(train_data.images.detach().cpu().numpy()))
mean_fluence = (np.mean(train_data.fluences.detach().cpu().numpy()))
std_fluence = (np.std(train_data.fluences.detach().cpu().numpy()))

for fold in [0, 1, 2, 3, 4]:
    if not os.path.exists(f"{BASE_PATH}{fold}/model_parameters.pt"):
        continue

    # ##############################################################################################################
    # Initialising network and loading trained weights
    # ##############################################################################################################
    device = torch.device("cpu")
    model = RegressionUNet(out_channels=2)
    model.load_state_dict(torch.load(f"{BASE_PATH}{fold}/model_parameters.pt"))
    model.to(device)
    model.float()

    if REEVALUATE_TEST_DATA or not os.path.exists(f"{BASE_PATH}{fold}/data.npz"):
        print("recomputing results...")

        # ##############################################################################################################
        # Analyse test data
        # ##############################################################################################################

        val_data = PalpaitineDataset(data_path=f"C:/final_data_simulation/test",
                                     use_all_data=True,
                                     train=False, device=device, fold=0,
                                     mean_musp=mean_musp,
                                     std_musp=std_musp,
                                     mean_mua=mean_mua,
                                     std_mua=std_mua,
                                     mean_signal=mean_signal,
                                     std_signal=std_signal,
                                     mean_fluence=mean_fluence,
                                     std_fluence=std_fluence,
                                     experimental_data=True)
        valloader = DataLoader(dataset=val_data, batch_size=1)

        gt_inputs = []
        gt_muas = []
        gt_segmentations = []
        est_muas = []
        est_fluences = []
        gt_fluences = []

        for v_i, (inputs, segmentation, mua, musp, fluence, instance_seg) in enumerate(valloader):
            print(f"{v_i}/{len(valloader)}")
            gt_inputs.append(inputs.detach().cpu().numpy())
            gt_muas.append(mua.detach().cpu().numpy())
            gt_segmentations.append(instance_seg.detach().cpu().numpy())
            gt_fluences.append(fluence.detach().cpu().numpy())

            outputs_mua = model(inputs)
            est_muas.append(outputs_mua[:, 0:1, :, :].detach().cpu().numpy())
            est_fluences.append(outputs_mua[:, 1:2, :, :].detach().cpu().numpy())

        np.savez(f"{BASE_PATH}{fold}/data.npz",
                 gt_inputs=(np.asarray(gt_inputs) * std_signal) + mean_signal,
                 gt_muas=(np.asarray(gt_muas) * std_mua) + mean_mua,
                 gt_segmentations=np.asarray(gt_segmentations),
                 est_muas=(np.asarray(est_muas) * std_mua) + mean_mua,
                 est_fluences=np.asarray(est_fluences) * std_fluence + mean_fluence,
                 gt_fluences=np.asarray(gt_fluences) * std_fluence + mean_fluence
                 )

    # ##############################################################################################################
    # Analyse Mouse data
    # ##############################################################################################################

    if REEVALUATE_MOUSE_DATA or not os.path.exists(f"{BASE_PATH}{fold}/mouse_data.npz"):
        mouse_data = PalpaitineMouseDataset(data_path=f"C:/final_data_simulation/mouse/",
                                            device=device,
                                            mean_signal=mean_signal,
                                            std_signal=std_signal)
        mouse_data_loader = DataLoader(dataset=mouse_data, batch_size=1)

        mouse_inputs = []
        mouse_est_muas = []
        mouse_est_fluences = []

        for v_i, inputs in enumerate(mouse_data_loader):
            print(f"{v_i}/{len(mouse_data_loader)}")
            mouse_inputs.append(inputs.detach().cpu().numpy())
            outputs_mua = model(inputs)
            mouse_est_muas.append(outputs_mua[:, 0:1, :, :].detach().cpu().numpy())
            mouse_est_fluences.append(outputs_mua[:, 1:2, :, :].detach().cpu().numpy())

        print("Data shape: ", np.shape(np.asarray(mouse_inputs)))

        np.savez(f"{BASE_PATH}{fold}/mouse_data.npz",
                 gt_inputs=(np.asarray(mouse_inputs) * std_signal) + mean_signal,
                 est_muas=(np.asarray(mouse_est_muas) * std_mua) + mean_mua,
                 est_fluences=(np.asarray(mouse_est_fluences) * std_fluence) + mean_fluence,
                 )

    # ##############################################################################################################
    # Analyse flow data
    # ##############################################################################################################

    if REEVALUATE_FLOW_DATA or not os.path.exists(f"{BASE_PATH}{fold}/flow_data.npz"):
        mouse_data = PalpaitineMouseDataset(data_path=f"C:/final_data_simulation/flow/",
                                            device=device,
                                            mean_signal=mean_signal,
                                            std_signal=std_signal)
        mouse_data_loader = DataLoader(dataset=mouse_data, batch_size=1)

        mouse_inputs = []
        mouse_est_muas = []
        mouse_est_fluences = []

        for v_i, inputs in enumerate(mouse_data_loader):
            print(f"{v_i}/{len(mouse_data_loader)}")
            mouse_inputs.append(inputs.detach().cpu().numpy())
            outputs_mua = model(inputs)
            mouse_est_muas.append(outputs_mua[:, 0:1, :, :].detach().cpu().numpy())
            mouse_est_fluences.append(outputs_mua[:, 1:2, :, :].detach().cpu().numpy())

        np.savez(f"{BASE_PATH}{fold}/flow_data.npz",
                 gt_inputs=(np.asarray(mouse_inputs) * std_signal) + mean_signal,
                 est_muas=(np.asarray(mouse_est_muas) * std_mua) + mean_mua,
                 est_fluences=(np.asarray(mouse_est_fluences) * std_fluence) + mean_fluence,
                 )
