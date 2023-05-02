import os
import torch
import numpy as np
from data_path import DATA_PATH
from torch.utils.data import DataLoader
from utils.networks import RegressionUNet
from utils.data_loading import PalpaitineDataset

NUM_EPOCHS = 200
BASE_PATH = rf"{DATA_PATH}/model_weights_simulation\fold_"

for fold in [0, 1, 2, 3, 4]:

    if not os.path.exists(f"{BASE_PATH}{fold}"):
        os.makedirs(f"{BASE_PATH}{fold}")

    train_data = PalpaitineDataset(data_path=f"{DATA_PATH}/training",
                                   augment=False, use_all_data=True, experimental_data=False)

    mean_musp = (np.mean(train_data.scatterings.detach().cpu().numpy()))
    std_musp = (np.std(train_data.scatterings.detach().cpu().numpy()))
    mean_mua = (np.mean(train_data.absorptions.detach().cpu().numpy()))
    std_mua = (np.std(train_data.absorptions.detach().cpu().numpy()))
    mean_signal = (np.mean(train_data.images.detach().cpu().numpy()))
    std_signal = (np.std(train_data.images.detach().cpu().numpy()))
    mean_fluence = (np.mean(train_data.fluences.detach().cpu().numpy()))
    std_fluence = (np.std(train_data.fluences.detach().cpu().numpy()))

    device = torch.device("cuda:0")
    train_data = PalpaitineDataset(data_path=f"{DATA_PATH}/training", train=True, device=device, fold=fold,
                                   augment=True,
                                   mean_signal=mean_signal,
                                   std_signal=std_signal,
                                   mean_mua=mean_mua,
                                   std_mua=std_mua,
                                   mean_musp=mean_musp,
                                   std_musp=std_musp,
                                   mean_fluence=mean_fluence,
                                   std_fluence=std_fluence,
                                   experimental_data=False)
    val_data = PalpaitineDataset(data_path=f"{DATA_PATH}/training", train=False, device=device, fold=fold,
                                 mean_signal=mean_signal,
                                 std_signal=std_signal,
                                 mean_mua=mean_mua,
                                 std_mua=std_mua,
                                 mean_musp=mean_musp,
                                 std_musp=std_musp,
                                 mean_fluence=mean_fluence,
                                 std_fluence=std_fluence,
                                 experimental_data=False
                                 )

    trainloader = DataLoader(dataset=train_data, shuffle=True, batch_size=15)
    valloader = DataLoader(dataset=val_data, shuffle=False, batch_size=10)
    num_iterations_per_epoch = len(trainloader)

    model = RegressionUNet(out_channels=2)
    model.to(device)

    # use half precision for faster training and smaller VRAM footprint.
    model.float()

    criterion_regression = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=10, factor=0.9)
    training_losses = np.zeros((NUM_EPOCHS * num_iterations_per_epoch, 4))
    validation_losses = np.zeros((NUM_EPOCHS, 4))

    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, segmentation, mua, _, fluence, _ = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs_mua = model(inputs)
            loss = criterion_regression(outputs_mua[:, 0:1], mua) + criterion_regression(outputs_mua[:, 1:2], fluence)
            loss.backward()
            training_losses[epoch * num_iterations_per_epoch + i] = [loss.item()]

            # Had exploding gradients in the early stages. This is a part of the fixes.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            running_loss += loss.item()
            if i % 25 == 24:
                print(f'[{epoch + 1} {i+1:5d}] train loss: {loss.item():.5f} ')

        running_val_loss = 0.0
        for v_i, v_data in enumerate(valloader, 0):
            inputs, segmentation, mua, _, fluence, instance_seg = v_data
            # forward + backward + optimize
            est_mua = model(inputs)
            v_loss = criterion_regression(est_mua[:, 0:1], mua) + criterion_regression(est_mua[:, 1:2], fluence)
            running_val_loss += v_loss.item()

        validation_losses[epoch] = [running_val_loss/len(valloader)]
        scheduler.step(running_val_loss/len(valloader))
        if not os.path.exists(f"{BASE_PATH}{fold}/progress/"):
            os.makedirs(f"{BASE_PATH}{fold}/progress/")

        np.savez(f"{BASE_PATH}{fold}/losses.npz",
                 training_losses=training_losses,
                 validation_losses=validation_losses)

        torch.save(model.state_dict(), f"{BASE_PATH}{fold}/model_parameters.pt")

        print(f'[{epoch + 1}] train loss: {running_loss/len(trainloader):.5f} '
              f'val_loss: {running_val_loss/len(valloader):.5f}')

    print('Finished training for fold')
print('Finished training completely')
