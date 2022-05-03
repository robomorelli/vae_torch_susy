import torch
from torch import nn
import math
from model.core import InverseSquareRootLinearUnit, ClippedTanh, SmashTo0, h1_prior, \
                        Dec1, CustomLinear, ConstrainedDec


class VAE(nn.Module):
    def __init__(self, original_dim, intermediate_dim,
                 latent_dim, nf_lognorm, nf_binomial):
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.Nf_lognorm = nf_lognorm
        self.Nf_binomial = nf_binomial

        super(VAE, self).__init__()
        self.enc = nn.Sequential(nn.Linear(self.original_dim, self.intermediate_dim), nn.ReLU(True)
                                 , nn.Linear(self.intermediate_dim, self.intermediate_dim), nn.ReLU(True))

        self.mu = nn.Linear(self.intermediate_dim, self.latent_dim)
        self.sigma = nn.Linear(self.intermediate_dim, self.latent_dim)

        #self.dec1 = nn.Linear(self.latent_dim, self.intermediate_dim)
        self.dec1 = Dec1(self.latent_dim, self.intermediate_dim) #equivalent to self.dec1 = ConstrainedDec(self.latent_dim, self.intermediate_dim)
        self.dec = nn.Sequential(nn.ReLU(True),
                                 nn.Linear(self.intermediate_dim, self.intermediate_dim), nn.ReLU(True))

        self.par1 = nn.Linear(self.intermediate_dim, self.original_dim)
        self.par2 = nn.Linear(self.intermediate_dim, self.Nf_lognorm)
        self.par3 = nn.Linear(self.intermediate_dim, self.Nf_lognorm)

        self.act2 = InverseSquareRootLinearUnit()
        self.act3 = ClippedTanh()

        self.SmashTo0 = SmashTo0()
        ######################################################
        #self.h1_prior = nn.Linear(self.original_dim, 1)
        #self.h1_prior.weight.data.fill_(0)
        #self.h1_prior.bias.data.fill_(1)

        ##################CUSTOM LAYER###############
        self.h1_prior = h1_prior(self.original_dim, 1)

        ##################CUSTOM LAYER###############
        ##################CUSTOM LAYER############### Defined to be skipped in init weight if isinstance(module, nn.Linear) is false
        self.mu_prior = CustomLinear(1,  latent_dim)
        self.mu_prior.weight.data.fill_(0)
        self.mu_prior.bias.data.fill_(0)

        ##################CUSTOM LAYER###############
        ##################CUSTOM LAYER############### Defined to be skipped in init weight if isinstance(module, nn.Linear) is false
        self.sigma_prior_preActivation = CustomLinear(1, latent_dim)
        self.sigma_prior_preActivation.weight.data.fill_(0)
        self.sigma_prior_preActivation.bias.data.fill_(1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            print(module)
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)

    #def _init_weights(self, module):
    #    if isinstance(module, nn.Linear):
    #        print(module)
            #module.weight.data.normal_(mean=0, std=1)

            #if module.bias is not None:
            #    module.bias.data.zero_()

    def encode(self, x):
        enc = self.enc(x)
        mu = self.mu(enc)
        sigma_pre = self.sigma(enc)

        fixed_input = self.SmashTo0(x)
        #with torch.no_grad():
        #    h1_prior = self.h1_prior(fixed_input)
        h1_prior_x = self.h1_prior(fixed_input)

        mu_prior = self.mu_prior(h1_prior_x)
        sigma_prior_preActivation = self.sigma_prior_preActivation(h1_prior_x)
        sigma_prior = self.act2(sigma_prior_preActivation)

        return mu, self.act2(sigma_pre), mu_prior, sigma_prior

    @staticmethod
    def sample(mu, sigma):
        std = sigma
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.view(z.size(0), self.latent_dim)
        d = self.dec1(z)
        d = self.dec(d)
        par2 = self.par2(d)
        par3 = self.par3(d)
        return self.par1(d), self.act2(par2), self.act3(par3)

    def forward(self, x):
        mu, sigma, mu_prior, sigma_prior = self.encode(x.view(x.size(0), self.original_dim))
        z = self.sample(mu, sigma)

        return self.decode(z), mu, sigma, mu_prior, sigma_prior
