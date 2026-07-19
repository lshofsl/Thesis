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

def slow_perception(rgba, hidden):   #Here we take the NCA channels and compute the local input of the slow controller
    # v: RGBA, h 2 first hidden channels 
    alpha = rgba[:, 3:4, :, :] # Extract ONLY the alpha channel
    h_layers = hidden[:, 0:2, :, :]

    eroded = -torch.nn.functional.max_pool2d(-alpha, kernel_size=3, stride=1, padding=1)
    edges = alpha - eroded

    alpha_padded = torch.nn.functional.pad(alpha, [1,1,1,1], mode='circular')
    lap_alpha = torch.nn.functional.conv2d(alpha_padded, lap_kernel, padding=0)

    # Q has 5 channels: [alpha, edges, lap, h1, h2]
    Q = torch.cat([alpha, edges, lap_alpha, h_layers], dim=1)
    return Q



class NCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, recurrent =3, modulatory=3):
        super().__init__()
        self.chn = chn
        public = chn - recurrent - modulatory  # NCA update only the RGBA+hidden channels but perceives all the channles except RA and modulatory gene channels

        
        dummy = torch.zeros([1, public, 8, 8], device="cuda:0")
        perc_chn = reduced_perception(dummy, 0).shape[1]
    
        self.w1 = torch.nn.Conv2d(perc_chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, public, 1, bias=False)  #Only for RGBA+hidden channels 
        self.w2.weight.data.zero_()
        
        
        #Parameter of the RA 
        self.alpha = torch.nn.Parameter(torch.tensor(0.1)) # Decay rate of the activator/phase
        self.beta  = torch.nn.Parameter(torch.tensor(0.1)) # Decay rate of the inhibitor/injury
        self.omega = torch.nn.Parameter(torch.tensor(0.0)) # Angular drift
        self.K     = torch.nn.Parameter(torch.tensor(0.5)) # Diffusion strength
        self.kappa = torch.nn.Parameter(torch.tensor(0.5)) # Spatial coupling between activator and inhibitor
        self.dt    = 0.1

        # Inputs for the slow perception of the RA 
        # Q -> Ia, Ib, Id
        self.slow_input_net = torch.nn.Conv2d(5, 3, kernel_size=1)
        # Translation from the RA state to the gene modulation output
        # a,b,d -> m_g, m_s, m_r
        
        #self.modulator_net = torch.nn.Conv2d(3, 3, kernel_size=1)
        #self.mod_proj = torch.nn.Conv2d(3, hidden_n, 1)   # Projection of the RA modulation into the hidden space of the NCA network
        #torch.nn.init.normal_(self.mod_proj.weight, std=0.01)
        #torch.nn.init.zeros_(self.mod_proj.bias)  #Initialization near of zero of the modulation projection to avoid instabilities at the beginning of training


        #FiLM modulation
        self.mod_gamma = torch.nn.Conv2d(3, hidden_n, 1)
        self.mod_beta  = torch.nn.Conv2d(3, hidden_n, 1)
        
        
        torch.nn.init.zeros_(self.mod_gamma.weight)
        torch.nn.init.zeros_(self.mod_gamma.bias)

        torch.nn.init.normal_(self.mod_beta.weight, std=0.01)
        torch.nn.init.zeros_(self.mod_beta.bias)
        
        
    def forward(self, x, update_rate=0.5,  step=0, k=4):
        #Initialize variables from x
        prefix = x[:, :16, ...].clone()    # RGBA + Hidden
        a = x[:, 16:17].clone()
        b = x[:, 17:18].clone()
        d = x[:, 18:19].clone()
        mod = x[:, 19:22].clone()


        # Phase/Amplitude initialization
        #phase, amplitude = ring_attractor_phases(a, b)

        # Slow RA updates
        if step % k == 0 : # Update the RA every k steps (including the first step)
            Q = slow_perception(x[:, :4], x[:, 4:16]) 
            I_signals = self.slow_input_net(Q)
            Ia, Ib, Id = I_signals[:, 0:1], I_signals[:, 1:2], I_signals[:, 2:3]
            
            new_a, new_b, new_d = discrete_update(
                a, b, d, self.alpha, self.beta, self.omega, 
                self.kappa, self.K, Ia, Ib, Id, dt=self.dt
            )
            new_a, new_b = consensus_update(new_a, new_b, dt=self.dt, mode='local')

            # Use of the new RA states to compute the modulation for the gene propagation
            a, b, d = new_a, new_b, new_d
            
        ra_stack = torch.cat([a, b, d], dim=1)  # Final a,b,d states after the RA dynamics evolution 
        gamma = 1.0 + torch.tanh(self.mod_gamma(ra_stack))
        beta  = torch.tanh(self.mod_beta(ra_stack))


        # 3. Fast NCA Logic
        fast_input = reduced_perception(x[:, :public], 0) # We only use the RGBA + hidden for the fast perception
        h = self.w1(fast_input)
        h = gamma * h + beta
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
            new_public, # 0:16
            a,          # 16
            b,          # 17
            d,          # 18
            mod         # 19:22
        ], dim=1)

        phase, amplitude = ring_attractor_phases(a, b)
        return x_final, phase, amplitude





