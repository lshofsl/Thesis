from multiprocessing import dummy
from sys import prefix

import torch

def perchannel_conv(x, filters):
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda:0")
ones = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda:0")
sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device="cuda:0")
lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device="cuda:0")
gaus = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device="cuda:0")


def perception(x, mask_n=0):

    filters = torch.stack([sobel_x, sobel_x.T, lap])
    if mask_n != 0:
        n = x.shape[1]
        padd = torch.zeros((x.shape[0], 3 * mask_n, x.shape[2], x.shape[3]), device="cuda:0")
        obs = perchannel_conv(x[:, 0:n - mask_n], filters)
        return torch.cat((x, obs, padd), dim=1)
    else:
        obs = perchannel_conv(x, filters)
        return torch.cat((x,obs), dim = 1 )

def masked_perception(x, mask_n=0):

    filters = torch.stack([sobel_x, sobel_x.T, lap])
    mask = torch.zeros_like(x)
    mask[:,0:x.shape[1]- mask_n,...] = 1
    x_masked = x*mask


    obs = perchannel_conv(x_masked,filters)
    return torch.cat((x,obs), dim = 1 )


def reduced_perception(x, mask_n=0):

    filters = torch.stack([sobel_x, sobel_x.T, lap])
    x_redu = x[:,0:x.shape[1]-mask_n]
    obs = perchannel_conv(x_redu,filters)
    return torch.cat((x,obs), dim = 1 )

class DummyVCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, mask_n=0):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(4 * chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.mask_n = mask_n

    def forward(self, x, update_rate=0.5):
        y = perception(x, self.mask_n)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp  = torch.nn.functional.pad(x[:, None, 3, ...],pad = [1,1,1,1] ,mode= "circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0,).cuda() > 0.1
        # Perform update
        x = x + y * update_mask * pre_life_mask
        return x

class MaskedCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, mask_n=0):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(4 * chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.mask_n = mask_n

    def forward(self, x, update_rate=0.5):
        y = masked_perception(x, self.mask_n)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp  = torch.nn.functional.pad(x[:, None, 3, ...],pad = [1,1,1,1] ,mode= "circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0).cuda() > 0.1
        # Perform update
        x = x + y * update_mask * pre_life_mask
        return x


class ReducedCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, mask_n=0):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn + 3*(chn-  mask_n), hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.mask_n = mask_n

    def forward(self, x, update_rate=0.5):
        y = reduced_perception(x, self.mask_n)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp  = torch.nn.functional.pad(x[:, None, 3, ...],pad = [1,1,1,1] ,mode= "circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0,).cuda() > 0.1
        # Perform update
        x = x + y * update_mask * pre_life_mask
        return x


#Slow RA functions 
#In each cell of the NCA we are going to add the RA states this will help us to understand the dynamics of training 

#Laplacian Kernel
lap_kernel = torch.tensor([[1.0, 2.0, 1.0], 
                           [2.0, -12., 2.0], 
                           [1.0, 2.0, 1.0]], dtype=torch.float32, device="cuda:0")
lap_kernel = (lap_kernel / 12.0).view(1, 1, 3, 3) # Normalization 

def ring_attractor_phases(a, b):
    local_amplitude = torch.sqrt(a**2 + b**2 + 1e-6)
    local_angle = torch.atan2(b, a)
    return local_amplitude, local_angle

def discrete_update(a, b, d, alpha, beta, omega, kappa, K, I_a, I_b, I_d, dt): 

    a_padded = torch.nn.functional.pad(a, [1,1,1,1], mode='circular')  #Mantain the circular shape in all the network 
    diff_a = torch.nn.functional.conv2d(a_padded, lap_kernel, padding=0)
    new_a = a + dt * (-alpha * a + omega * b + K * diff_a + I_a)
    
    b_padded = torch.nn.functional.pad(b, [1,1,1,1], mode='circular')
    diff_b = torch.nn.functional.conv2d(b_padded, lap_kernel, padding=0)
    new_b = b + dt * (-alpha * b - omega * a + K * diff_b + I_b)
    
    d_padded = torch.nn.functional.pad(d, [1,1,1,1], mode='circular')
    diff_d = torch.nn.functional.conv2d(d_padded, lap_kernel, padding=0)
    new_d = d + dt * (-beta * d + kappa * diff_d + I_d)
    
    return new_a, new_b, new_d

def consensus_update(a, b, dt, mode='local'):
    if mode == 'local':
        a_avg = torch.nn.functional.avg_pool2d(a, 5, 1, 2)   # Kuramoto-like local averaging for phase synchronization
        b_avg = torch.nn.functional.avg_pool2d(b, 5, 1, 2)
    else:
        a_avg = torch.mean(a, dim=(2, 3), keepdim=True)
        b_avg = torch.mean(b, dim=(2, 3), keepdim=True)

    # Normalization over the average to maintain the amplitude of the local state
    rho_avg = torch.sqrt(a_avg**2 + b_avg**2 + 1e-6)
    rho_local = torch.sqrt(a**2 + b**2 + 1e-6)
    
    # Consensus update with amplitude normalization
    a_avg_norm = (a_avg / rho_avg) * rho_local
    b_avg_norm = (b_avg / rho_avg) * rho_local

    a = a + dt * (a_avg_norm - a)
    b = b + dt * (b_avg_norm - b)
    return a, b

def slow_perception(channels):
    # For the GeneCA we need to sharper the boundaries so the RA can have an exploratory phase (as GenePropCA). By doing this the living mask
    # pass from a continous state to a discrete one, we do this because we work with unnitilize state which in different of GenePropCA that works
    # with the weights that the GeneCA obtain, here we work with zero weights.
    alpha = channels[:, 3:4, :, :] # Extract ONLY the alpha channel
    h_layers = channels[:, 4:6, :, :]
    genes = channels[:,13:16,:, :]
    
    alpha_sharp = torch.sigmoid((alpha - 0.5) * 10.0) 
    
    eroded = -torch.nn.functional.max_pool2d(-alpha_sharp, kernel_size=3, stride=1, padding=1)
    edges = alpha_sharp - eroded
    
    alpha_padded = torch.nn.functional.pad(alpha_sharp, [1,1,1,1], mode='circular')
    lap_alpha = torch.nn.functional.conv2d(alpha_padded, lap_kernel, padding=0)

    Q = torch.cat([alpha_sharp, edges, lap_alpha, h_layers, genes], dim=1)
    return Q



class GeneCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size=3, recurrent_gene =3, modulatory_gene=3):
        super().__init__()
        self.public = chn - gene_size - recurrent_gene - modulatory_gene  # GeneNCA update only the RGBA+hidden channels but perceives all the channles except RA and modulatory gene channels
        self.private = gene_size 
        self.fast_channels = self.public + self.private
        self.w1 = torch.nn.Conv2d(self.fast_channels + 3 * (self.fast_channels), hidden_n, 1) 
        self.w2 = torch.nn.Conv2d(hidden_n, self.public, 1, bias=False)
        self.w2.weight.data.zero_()
        
        #Parameter of the RA 
        self.alpha = torch.nn.Parameter(torch.tensor(0.1)) # Decay rate of the activator/phase
        self.beta  = torch.nn.Parameter(torch.tensor(0.1)) # Decay rate of the inhibitor/injury
        self.omega = torch.nn.Parameter(torch.tensor(0.1)) # Angular drift
        self.K     = torch.nn.Parameter(torch.tensor(0.1)) # Diffusion strength
        self.kappa = torch.nn.Parameter(torch.tensor(0.1)) # Spatial coupling between activator and inhibitor
        self.dt    = 0.1

        # Inputs for the slow perception of the RA 
        # Q -> Ia, Ib, Id
        self.slow_input_net = torch.nn.Conv2d(8, 3, kernel_size=1)
        # Translation from the RA state to the gene modulation output
        # a,b,d -> m_g, m_s, m_r
        self.modulator_net = torch.nn.Conv2d(3, 3, kernel_size=1)
        self.mod_proj = torch.nn.Conv2d(3, hidden_n, 1)   # Projection of the RA modulation into the hidden space of the NCA network
        torch.nn.init.normal_(self.mod_proj.weight, std=0.01)
        torch.nn.init.zeros_(self.mod_proj.bias)  #Initialization near of zero of the modulation projection to avoid instabilities at the beginning of training


    def forward(self, x, update_rate=0.5,  step=0, k=4):
        #Initialize variables from x
        prefix = x[:, :13, ...].clone()    # RGBA + Hidden
        gene = x[:, 13:16, ...].clone()      # Gene Encoding
        a = x[:, 16:17].clone()
        b = x[:, 17:18].clone()
        d = x[:, 18:19].clone()
        mod = x[:, 19:22].clone()


        # Phase/Amplitude initialization
        phase, amplitude = ring_attractor_phases(a, b)


        # Slow RA updates
        if step % k == 0 and step >=20 : 
            Q = slow_perception(x[:, :16])  # consider adding gene channels here
            I_signals = self.slow_input_net(Q)
            Ia, Ib, Id = I_signals[:, 0:1], I_signals[:, 1:2], I_signals[:, 2:3]
            new_a, new_b, new_d = discrete_update(a, b, d, self.alpha, self.beta, self.omega,
                                               self.kappa, self.K, Ia, Ib, Id, dt=self.dt)
            new_a, new_b = consensus_update(new_a, new_b, dt=self.dt, mode='local')
            a, b, d = new_a, new_b, new_d

        ra_stack = torch.cat([a, b, d], dim=1)
        raw_mod = self.modulator_net(ra_stack)
        mod_term = ra_strength * torch.tanh(self.mod_proj(raw_mod))


        # 3. Fast NCA Logic
        fast_input = reduced_perception(x[:, :16], 0) # We only use the RGBA + Gene for the fast perception, not the RA states
        h = self.w1(fast_input)          
        h = h + mod_term        # We project the RA modulation into the hidden space. We do this as we work with 2 time scales, the RA modulation should affect the hidden representation of the NCA before the output layer.
        y = self.w2(torch.relu(h)) 
        
        # Masks
        b_sz, c_sz, h, w = y.shape
        update_mask = (torch.rand(b_sz, 1, h, w, device=x.device) + update_rate).floor()
        xmp = torch.nn.functional.pad(x[:, 3:4, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = (torch.nn.functional.max_pool2d(xmp, 3, 1, 0) > 0.1).to(x.device)


        #delta update 
        delta = y * update_mask * pre_life_mask.to(y.dtype)

        #  Update of the new public channels (prefix)
        new_public =  prefix + delta 
        # We concatenate all parts to create x_final without ever modifying the input x
        x_final = torch.cat([
            new_public, # 0:13
            gene,       # 13:16
            a,          # 16
            b,          # 17
            d,          # 18
            mod         # 19:22
        ], dim=1)


        phase, amplitude = ring_attractor_phases(a, b)
        return x_final, phase, amplitude





