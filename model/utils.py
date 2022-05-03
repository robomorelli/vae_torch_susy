import numpy as np
import torch

clip_x_to0 = 1e-4

def SmashTo0(x):
    return 0*x

def loss_function(x, pars, Nf_lognorm, Nf_binomial, weights_loss):
    recon_loss = RecoProb_forVAE(x, pars[0], pars[1], pars[2], Nf_lognorm, Nf_binomial, weights_loss)
    return recon_loss

def RecoProb_forVAE(x, par1, par2, par3, Nf_lognorm, Nf_binomial, weights_loss = [1, 1, 1]):

    N = 0
    nll_loss = 0

    lognorm_weights = [x for x in weights_loss[:Nf_lognorm]]
    binomail_weights = [x for x in weights_loss[:Nf_lognorm]]

    #Log-Normal distributed variables
    for ix, wi in enumerate(lognorm_weights):

        mu = par1[:,ix:ix+1]
        sigma = par2[:,ix:ix+1]
        fraction = par3[:,ix:ix+1]

        x_clipped = torch.clamp(x[:,ix:ix+1], clip_x_to0, 1e8)
        single_NLL = torch.where(torch.le(x[:,ix:ix+1], clip_x_to0),
                                -torch.log(fraction),
                                    -torch.log(1-fraction)
                                    + torch.log(sigma)
                                    + torch.log(x_clipped)
                                    + 0.5*torch.mul(torch.div(torch.log(x_clipped) - mu, sigma),
                                                      torch.div(torch.log(x_clipped) - mu, sigma)))
        nll_loss += torch.sum(wi*single_NLL, axis=-1)

    N += Nf_lognorm

    #Binomial distributed variables

    for ix, wi in enumerate(binomail_weights):
        p = 0.5*(1+0.98*torch.tanh(par1[:, N+ix: N+ix+Nf_binomial]))
        single_NLL = -torch.where(torch.eq(x[:, N: N+Nf_binomial],1), torch.log(p), torch.log(1-p))
        nll_loss += torch.sum(wi*single_NLL, axis=-1)

    N += Nf_binomial

    return nll_loss

def KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior):
    kl_loss = torch.mul(torch.mul(sigma, sigma), torch.mul(sigma_prior, sigma_prior))
    div = torch.div(mu_prior - mu, sigma_prior)
    kl_loss += torch.mul(div, div)
    kl_loss += torch.log(torch.div(sigma_prior, sigma)) -1
    return 0.5 * torch.sum(kl_loss, axis=-1)

#def KL_loss_forVAE(mu, sigma):
#    mu_prior = torch.tensor(0)
#    sigma_prior = torch.tensor(1)
#    kl_loss = torch.mul(torch.mul(sigma, sigma), torch.mul(sigma_prior,sigma_prior))
#    div = torch.div(mu_prior - mu, sigma_prior)
#    kl_loss += torch.mul(div, div)
#    kl_loss += torch.log(torch.div(sigma_prior, sigma)) -1
#    return 0.5 * torch.sum(kl_loss, axis=-1)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
