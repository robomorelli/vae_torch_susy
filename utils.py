#!/usr/bin/env python3
import pandas as pd
import numpy as np
import uproot
import random
from sklearn.model_selection import train_test_split
from ray import tune
from torch.utils.data import DataLoader
from model.utils import KL_loss_forVAE, loss_function, EarlyStopping
from model.model import VAE
from utils import *
from config import *
from dataset.utils import *

def cutflow_bkg(data_folder, train_sample_name, depth, additional_cuts):

    entrysteps=3000000
    tot = 0

    # for n in TrainSamplesName:
    print('processing {}[{}]'.format(data_folder, train_sample_name))

    events = uproot.open(data_folder)[train_sample_name]
    name = train_sample_name.split('_')[0]
    array = events.lazyarray('met')

    print('numero di eventi:', len(array))
    file_split = len(array)//entrysteps
    start_name_file = 0
    entrystart = start_name_file*entrysteps

    batches = events.iterate(columns, entrystart=entrystart,
                           entrysteps=entrysteps,
                               outputtype=pd.DataFrame)

    for ix in range(start_name_file, file_split+1):

        print('I splitted the root file: processsing {}/{} of {}'.format(ix+1, file_split+1, name))
        batch = next(batches)
        print('adding luminosity')
        batch['luminosity'] = 139000
        print(len(batch))

        batch = batch[batch['nLep_signal'].astype(int)==1]
        print('after signal {}'.format(len(batch)))

        batch = batch[batch['trigMatch_metTrig'].astype(int)==1]
        print('after trig {}'.format(len(batch)))

        batch = batch[(batch['jet2Pt']>30) | (batch['jet3Pt']>30)]
        batch = batch[batch['jet4Pt']<30]
#         batch = batch[((batch['jet2Pt']>30)|(batch['jet3Pt']>30))]
        print('after jetpt {}'.format(len(batch)))

        batch = batch[((batch['nBJet30_MV2c10']>=1)&(batch['nBJet30_MV2c10']<4))]
#         batch = batch[batch['nBJet30_MV2c10']==2]
        print('after bjet {}'.format(len(batch)))

        batch = batch[batch['met']>220]
        print('after met {}'.format(len(batch)))

        batch = batch[batch['mt']>50]
        print('after mt {}'.format(len(batch)))

        if depth == 'middle':
            batch = batch[((batch['mbb']>=100)&(batch['mbb']<=140))]
            print('after mbb {}'.format(len(batch)))
            batch = batch[batch['mct2']>100]
            print('after mct2 {}'.format(len(batch)))

        if additional_cuts:

            print('cutting below 0 and above 1000')
            batch = batch[((batch['mct2']>=0)&(batch['mct2']<1000))]
            batch = batch[((batch['mt']>=0)&(batch['mt']<1000))]
            batch = batch[((batch['met']>=0)&(batch['met']<1000))]
            batch = batch[((batch['mlb1']>=0)&(batch['mlb1']<1000))]
            batch = batch[((batch['lep1Pt']>=0)&(batch['lep1Pt']<1000))]

        if len(batch) > 0:

            batch['weight'] = batch.apply(lambda row: row['genWeight']*row['eventWeight']*row['pileupWeight']*
                                 row['leptonWeight']*row['bTagWeight']*row['jvtWeight']*row['luminosity'], axis=1)

            batch_fin = batch.iloc[:,:8]
            batch_fin['weight'] = batch['weight']

            batch_fin = batch_fin[['met', 'mt', 'mbb', 'mct2',
            'mlb1','lep1Pt', 'nJet30', 'nBJet30_MV2c10', 'weight']]

            tot = tot + len(batch)
            print('tot = {}'.format(tot))
            print("\x1b[31m\"saving {}_{}""\x1b[0m".format(name,ix))

            np.save(splitted_numpy_bkg + '/background_{}_{}.npy'.format(name,ix), batch_fin.values)

    return


def cutflow_sig(data_folder, train_sample_name, depth,additional_cuts):

    entrysteps=3000000
    tot = 0

    # for n in TrainSamplesName:
    print('processing {}[{}]'.format(data_folder, train_sample_name))

    events = uproot.open(data_folder)[train_sample_name]
    name = '_'.join(train_sample_name.split('_')[1:-1])
    array = events.lazyarray('met')

    print('numero di eventi:', len(array))
    file_split = len(array)//entrysteps
    start_name_file = 0
    entrystart = start_name_file*entrysteps

    batches = events.iterate(columns_sig, entrystart=entrystart,
                           entrysteps=entrysteps,
                               outputtype=pd.DataFrame)

    for ix in range(start_name_file, file_split+1):

        print('I splitted the root file: processsing {}/{} of {}'.format(ix+1, file_split+1, name))
        batch = next(batches)
        print('adding luminosity')
        batch['luminosity'] = 139000
        print(len(batch))

        batch = batch[batch['nLep_signal'].astype(int)==1]
        print('after signal {}'.format(len(batch)))

        batch = batch[batch['trigMatch_metTrig'].astype(int)==1]
        print('after trig {}'.format(len(batch)))

        batch = batch[((batch['nBJet30_MV2c10']>=1)&(batch['nBJet30_MV2c10']<4))]
#         batch = batch[batch['nBJet30_MV2c10']==2]
        print('after bjet {}'.format(len(batch)))

        batch = batch[batch['met']>220]
        print('after met {}'.format(len(batch)))

        batch = batch[batch['mt']>50]
        print('after mt {}'.format(len(batch)))

        if depth == 'middle':
            batch = batch[((batch['mbb']>=100)&(batch['mbb']<=140))]
            print('after mbb {}'.format(len(batch)))
            batch = batch[batch['mct2']>100]
            print('after mct2 {}'.format(len(batch)))

        if additional_cuts:
            print('cutting below 0 and above 1000')
            batch = batch[((batch['mct2']>=0)&(batch['mct2']<1000))]
            batch = batch[((batch['mt']>=0)&(batch['mt']<1000))]
            batch = batch[((batch['met']>=0)&(batch['met']<1000))]
            batch = batch[((batch['mlb1']>=0)&(batch['mlb1']<1000))]
            batch = batch[((batch['lep1Pt']>=0)&(batch['lep1Pt']<1000))]

        if len(batch) > 0:

            batch['weight'] = batch.apply(lambda row: row['genWeight']*row['eventWeight']*row['pileupWeight']*
                                 row['leptonWeight']*row['bTagWeight']*row['jvtWeight']*row['luminosity'], axis=1)

            batch_fin = batch.iloc[:,:8]
            batch_fin['weight'] = batch['weight']

            batch_fin = batch_fin[['met', 'mt', 'mbb', 'mct2',
            'mlb1','lep1Pt', 'nJet30', 'nBJet30_MV2c10', 'weight']]

            tot = tot + len(batch)
            print('tot = {}'.format(tot))
            print("\x1b[31m\"saving {}_{}""\x1b[0m".format(name,ix))

            np.save(numpy_sig + '/{}_{}.npy'.format(name,ix), batch_fin.values)

    return


def concatenate_file(input_data_folder, output_data_folder, data_kind):
    fh_name = os.listdir(input_data_folder)

    for i, name in enumerate(fh_name):
        if i == 0:
            joined_file = np.load(input_data_folder + '{}'.format(name))

        else:
            new_to_join = np.load(input_data_folder + '{}'.format(name))
            joined_file = np.concatenate((joined_file, new_to_join))

    np.save(output_data_folder + '{}.npy'.format(data_kind), joined_file)
    return

def train_val_test_split(input_data_folder, output_data_folder, proportions, seed, random_state):

    fh_name = os.listdir(input_data_folder)
    back_tot = np.load(input_data_folder + '{}'.format(fh_name[0]))

    random.seed(seed)

    y = [0 for x in range(len(back_tot))]

    if len(proportions) == 2:

        X_train, X_val, _, _ = train_test_split(back_tot, y, test_size=proportions[1], random_state=42)

        np.save(output_data_folder + '/background.npy', back_tot)
        np.save(output_data_folder + '/background_train.npy', X_train)
        np.save(output_data_folder + '/background_val.npy', X_val)

    elif len(proportions) == 3:

        X_train, X_val_test, _, _ = train_test_split(back_tot, y, test_size=proportions[1]+proportions[2], random_state=random_state)

        y = [0 for x in range(len(X_val_test))]

        print('train {}'.format(len(X_train)))

        print('len val + test {}'.format(len(y)))

        remaining_fraction = 1 - proportions[0]
        factor = proportions[2]/remaining_fraction

        print('factor is {} and test is {}'.format(factor, remaining_fraction*factor))

        X_val, X_test, _, _ = train_test_split(X_val_test, y, test_size=factor, random_state=42)

        print('val {}'.format(len(X_val)))
        print('test {}'.format(len(X_test)))

        np.save(output_data_folder + '/background.npy', back_tot)
        np.save(output_data_folder + '/background_train.npy', X_train)
        np.save(output_data_folder + '/background_val.npy', X_val)
        np.save(output_data_folder + '/background_test.npy', X_test)

    else:
        raise Exception('error: no splitting')

    print('train val and test ratio: {} {} {}'.format(proportions[0],proportions[1],proportions[2]))

    return

def load_data_train_eval(feat_selected, config, data_folder = train_val_test):

    bkg = np.load(data_folder + 'background_train.npy')
    bkg_val = np.load(data_folder + 'background_val.npy')

    train_df = pd.DataFrame(bkg[:, :-1], columns=cols_train)
    val_df = pd.DataFrame(bkg_val[:, :-1], columns=cols_train)

    train_data = PandasDF(train_df, feat_selected)
    val_data = PandasDF(val_df, feat_selected)

    batch_size = config['batch_size']

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_data, trainloader, valloader

def initialize_model(config, input_size):

    Nf_lognorm = config['Nf_lognorm']
    hidden_size = config['hidden_size']
    latent_dim = config['latent_dim']
    #act_fun = train_dict['act_fun'] #### TO DO: ALLOW for this coiche in the model buildign

    Nf_binomial = input_size - Nf_lognorm

    vae = VAE(input_size, hidden_size
              , latent_dim, Nf_lognorm, Nf_binomial)

    return vae

def train_class(config, vae, input_size, optimizer, trainloader, valloader,
                device, checkpoint_dir = None):

    cuda = torch.cuda.is_available()
    if cuda:
        print('added visible gpu')
        ngpus = torch.cuda.device_count()

    Nf_lognorm = config['Nf_lognorm']
    weight_KL_loss = config['weight_KL_loss']
    epochs = config['epochs']
    lr = config['lr']
    model_name = config['model_name']
    Nf_binomial = input_size - Nf_lognorm

    ####Train Loop####
    """
    Set the model to the training mode first and train
    """
    train_loss = []
    weights_loss = [5, 10, 10]
    patience = 1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',  factor=0.8, patience=patience, threshold=0.0001,
                                              threshold_mode='rel', cooldown=0, min_lr=9e-8, verbose=True)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "model_name"))
        vae.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(epochs):
        vae.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            x = data[0].to(device)
            pars, mu, sigma, mu_prior, sigma_prior = vae(x)

            recon_loss = loss_function(x, pars, Nf_lognorm,
                                       Nf_binomial, weights_loss).mean()

            KLD = KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior).mean()
            loss = recon_loss + weight_KL_loss * KLD  # the mean of KL is added to the mean of MSE
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loss.append(loss.item())

            if i % 10 == 0:
                print("Loss: {}".format(loss.item()))
                print("kl div {}".format(KLD))

        ###############################################
        # eval mode for evaluation on validation dataset
        ###############################################

        # Validation loss
        #val_loss = 0.0
        temp_val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():

                x = data[0].to(device)
                pars, mu, sigma, mu_prior, sigma_prior = vae(x)
                recon_loss = loss_function(x, pars, Nf_lognorm,
                                           Nf_binomial, weights_loss).mean()

                KLD = KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior).mean()
                temp_val_loss += recon_loss + weight_KL_loss * KLD

                val_steps += 1
        val_loss = temp_val_loss / len(valloader)
        val_loss_cpu = val_loss.cpu().item()
        print('validation_loss {}'.format(val_loss_cpu))
        scheduler.step(val_loss)
        return val_loss_cpu


def train(config, checkpoint_dir=None):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print('added visible gpu')
        ngpus = torch.cuda.device_count()

    #device = "cpu"
    #########################LOAD DATA###################START############
    train_data, trainloader, valloader = load_data_train_eval(train_val_test, config, feat_selected=['met', 'mt', 'mct2'])
    #########################LOAD DATA###################END#######

    ########### LOAD MODEL /TRAIN/EVAL##############
    input_size = train_data[0][0].size()[0]
    Nf_lognorm = config['Nf_lognorm']
    weight_KL_loss = config['weight_KL_loss']
    epochs = config['epochs']
    lr = config['lr']
    model_name = config['model_name']
    Nf_binomial = input_size - Nf_lognorm

    vae = initialize_model(config, input_size).to(device)
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

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        vae.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    #early_stopping = EarlyStopping(patience=50)
    for epoch in range(epochs):
        vae.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            x = data[0].to(device)
            pars, mu, sigma, mu_prior, sigma_prior = vae(x)

            recon_loss = loss_function(x, pars, Nf_lognorm,
                                       Nf_binomial, weights_loss).mean()

            KLD = KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior).mean()
            loss = recon_loss + weight_KL_loss * KLD  # the mean of KL is added to the mean of MSE
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loss.append(loss.item())

            if i % 10 == 0:
                print("Loss: {}".format(loss.item()))
                print("kl div {}".format(KLD))

        ###############################################
        # eval mode for evaluation on validation dataset
        ###############################################

        # Validation loss
        val_loss = 0.0
        temp_val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():

                x = data[0].to(device)
                pars, mu, sigma, mu_prior, sigma_prior = vae(x)
                recon_loss = loss_function(x, pars, Nf_lognorm,
                                           Nf_binomial, weights_loss).mean()

                KLD = KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior).mean()
                temp_val_loss += recon_loss + weight_KL_loss * KLD

                val_steps += 1
        val_loss = temp_val_loss / len(valloader)
        val_loss_cpu = val_loss.cpu().item()
        print('validation_loss {}'.format(val_loss))
        #scheduler.step(val_loss)

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.

        #vae.eval()
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "vae.pth")
            torch.save((vae.state_dict(), optimizer.state_dict()), path)

        #tune.report(loss=(val_loss_cpu))

        #if temp_val_loss < val_loss:
        #    print('val_loss improved from {} to {}, saving model to {}' \
        #          .format(val_loss, loss, save_model_path))
        #    torch.save(vae.state_dict(), save_model_path / model_name)
        #    val_loss = temp_val_loss

        #early_stopping(temp_val_loss)
        #if early_stopping.early_stop:
        #    break
