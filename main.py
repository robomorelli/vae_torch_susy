from torch.utils.data import Dataset
import torch.utils.data
from torch import optim
import numpy as np
from pathlib import Path
from model.model import VAE
from model.utils import KL_loss_forVAE, loss_function, EarlyStopping
from dataset.utils import *
from config import *
import random as rn

np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)


def main(save_model_path, **kwargs):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print('added visible gpu')
        ngpus = torch.cuda.device_count()

    bkg = np.load(train_val_test + '/background_train.npy')
    bkg_val = np.load(train_val_test + '/background_val.npy')

    train_df = pd.DataFrame(bkg[:, :-1], columns=cols_train)
    val_df = pd.DataFrame(bkg_val[:, :-1], columns=cols_train)

    train_data = PandasDF(train_df, ['met', 'mt', 'mct2'])
    val_data = PandasDF(val_df, ['met', 'mt', 'mct2'])

    batch_size = train_dict['batch_size']

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)

    ####initialize model####
    input_size = train_data[0][0].size()[0]

    Nf_lognorm = train_dict['Nf_lognorm']

    hidden_size = train_dict['hidden_size']
    latent_dim = train_dict['latent_dim']
    weight_KL_loss = train_dict['weight_KL_loss']
    epochs = train_dict['epochs']
    lr = train_dict['lr']
    #act_fun = train_dict['act_fun'] #### TO DO: ALLOW for this coiche in the model buildign
    model_name = train_dict['model_name']

    Nf_binomial = input_size - Nf_lognorm

    vae = VAE(input_size, hidden_size
              , latent_dim, Nf_lognorm, Nf_binomial).to(device)

    ####Train Loop####
    """
    Set the model to the training mode first and train
    """

    train_loss = []
    weights_loss = [5, 10, 10]
    val_loss = 10 ** 16
    patience = 1
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',  factor=0.8, patience=patience, threshold=0.0001,
                                              threshold_mode='rel', cooldown=0, min_lr=9e-8, verbose=True)
    early_stopping = EarlyStopping(patience=50)
    for epoch in range(epochs):
        vae.train()
        for i, data in enumerate(trainloader):

            optimizer.zero_grad()
            x = data[0].to(device)
            pars, mu, sigma, mu_prior, sigma_prior = vae(x)

            recon_loss = loss_function(x, pars, Nf_lognorm,
                                       Nf_binomial, weights_loss).mean()

            KLD = KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior).mean()
            loss = recon_loss + weight_KL_loss * KLD  # the mean of KL is added to the mean of MSE
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()

            if i % 10 == 0:
                print("Loss: {}".format(loss.item()))
                print("kl div {}".format(KLD))

        ###############################################
        # eval mode for evaluation on validation dataset
        ###############################################
        vae.eval()
        temp_val_loss = 0
        #with torch.no_grad():
        for i, data in enumerate(valloader):
            #data = (data.type(torch.FloatTensor)).to(device)
            x = data[0].to(device)
            pars, mu, sigma, mu_prior, sigma_prior = vae(x)
            recon_loss = loss_function(x, pars, Nf_lognorm,
                                       Nf_binomial, weights_loss).mean()

            KLD = KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior).mean()
            temp_val_loss += recon_loss + weight_KL_loss * KLD

        temp_val_loss = temp_val_loss / len(valloader)
        print('validation_loss {}'.format(temp_val_loss))
        scheduler.step(temp_val_loss)
        if temp_val_loss < val_loss:
            print('val_loss improved from {} to {}, saving model to {}' \
                  .format(val_loss, loss, save_model_path))
            torch.save(vae.state_dict(), save_model_path / model_name)
            val_loss = temp_val_loss


        early_stopping(temp_val_loss)
        if early_stopping.early_stop:
            break

if __name__ == "__main__":

    ###############################################
    # TO DO: add parser for parse command line args
    ###############################################
    train_dict = {"batch_size": 200,
        "hidden_size" : 20,
        "latent_dim" : 3,
        "weight_KL_loss" : 0.6,
        "Nf_lognorm" : 3,
        "epochs" : 100,
        "lr" : 0.003,
        "act_fun" : 'relu',
        "model_name" : 'vae_susy.h5'}

    save_model_path = Path(model_results_path)

    if not (save_model_path.exists()):
        print('creating path')
        os.makedirs(save_model_path)

    main(save_model_path, train_dict=train_dict)



