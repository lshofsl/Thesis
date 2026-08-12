from multiprocessing import dummy
from sys import prefix

import torch
import torch.nn as nn
import torch.nn.functional as F

ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda:0")
sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device="cuda:0")
sonel_y = sobel_x.T
lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device="cuda:0")
gaus = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device="cuda:0")

def perchannel_conv(x, filters):
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
    y = torch.nn.functional.conv2d(y, filters[:, None].to(x.device))
    return y.reshape(b, -1, h, w)


def perception(x, mask_n=0):

    filters = torch.stack([ident,sobel_x, sobel_x.T, lap])
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

class NCA(torch.nn.Module):
    def __init__(self, chn=16, hidden_n=64):
        super().__init__()
        self.chn = chn
        # Perception dimensionality: chn (identity) + 3 * chn (sobel x, sobel y, laplacian)
        self.w1 = torch.nn.Conv2d(chn + 3 * chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        torch.nn.init.zeros_(self.w2.weight)

    def get_alive_mask(self, x):
        alpha = x[:, 3:4, :, :] 
        padded_alpha = torch.nn.functional.pad(alpha, pad=[1, 1, 1, 1], mode="circular")
        return torch.nn.functional.max_pool2d(padded_alpha, 3, stride=1, padding=0) > 0.1

    def forward(self, x, update_rate=0.5):
        # Pre-life mask 
        pre_life_mask = self.get_alive_mask(x).to(x.dtype)
        
        # Perception & MLP forward pass
        y = reduced_perception(x, 0)
        y = self.w2(torch.relu(self.w1(y)))
        
        # Stochastic update mask
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device=x.device) < update_rate).to(x.dtype)
        
        # State step update gated by pre-life mask
        x = x + y * update_mask * pre_life_mask
        
        # Clean background using post-life mask cast to float
        post_life_mask = self.get_alive_mask(x).to(x.dtype)
        x = x * post_life_mask
        
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


# We apply the live_mask on the discrete update to no have a growth of a,b,d on the background
def discrete_update(a, b, d, mu, omega, beta_r, beta_i, beta_d, kappa, K,
                     I_a, I_b, I_d, dt, live_mask):
    # CGLE equations are used to as our Ring-Attracor imposition on the slow NCA, 
    # We look that by injecting this dynamics on the NCA manifold we can obtain a bette control

    a_padded = torch.nn.functional.pad(a, [1, 1, 1, 1], mode='circular')
    diff_a = torch.nn.functional.conv2d(a_padded, lap_kernel, padding=0)

    b_padded = torch.nn.functional.pad(b, [1, 1, 1, 1], mode='circular')
    diff_b = torch.nn.functional.conv2d(b_padded, lap_kernel, padding=0)

    r_sq = a**2 + b**2  # |z|^2, computed once, shared by both cubic terms

    cubic_a = -r_sq * (beta_r * a - beta_i * b)
    cubic_b = -r_sq * (beta_r * b + beta_i * a)

    new_a = a + dt * (mu * a - omega * b + cubic_a + K * diff_a + I_a)
    new_b = b + dt * (mu * b + omega * a + cubic_b + K * diff_b + I_b)

    # In addition of the coupled 2D system, we add an uncoupled diffusion equation to 
    # monitorate the dynamics 
    d_padded = torch.nn.functional.pad(d, [1, 1, 1, 1], mode='circular')
    diff_d = torch.nn.functional.conv2d(d_padded, lap_kernel, padding=0)
    new_d = d + dt * (-beta_d * d + kappa * diff_d + I_d)

    # Mask updates to live cells only to preserves organism boundary and is
    # what makes post-damage phase healing through K*diff_z observable
    new_a = new_a * live_mask + a * (1 - live_mask)
    new_b = new_b * live_mask + b * (1 - live_mask)
    new_d = new_d * live_mask + d * (1 - live_mask)

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


def slow_perception(rgba, hidden):
    alpha = rgba[:, 3:4, :, :]
    h_layers = hidden[:, 0:2, :, :]  # Ring attractor state channels

    # 1. Padding
    alpha_padded = torch.nn.functional.pad(alpha, [1,1,1,1], mode='circular')
    
    # 2. Convolutions with external filters
    lap_alpha = F.conv2d(alpha_padded, lap)
    lap_inward = -lap_alpha

    # 3. Gradients in x and y to have orientation 
    smooth_alpha = F.conv2d(alpha_padded, gaus)
    grad_x = F.conv2d(alpha_padded, sobel_x)
    grad_y = F.conv2d(alpha_padded, sobel_y)

    # 3. Morphological Boundary
    eroded = -F.max_pool2d(-alpha_padded, kernel_size=3, stride=1, padding=0)
    dilated = F.max_pool2d(alpha_padded, kernel_size=3, stride=1, padding=0)
    morph_edge = dilated - eroded


    Q = torch.cat([alpha, lap_inward, smooth_alpha, eroded, morph_edge, grad_x, grad_y, h_layers], dim=1)
    return Q




class NCA_RAMod(nn.Module):
    def __init__(self, chn=22, hidden_n=96, recurrent=3, modulatory=3):
        super().__init__()
        self.chn = chn
        self.public = chn - recurrent - modulatory  # Public channels = 16

        dummy = torch.zeros([1, self.public, 8, 8], device="cuda:0")
        perc_chn = reduced_perception(dummy, 0).shape[1]

        # Standard NCA Fast Network
        self.w1 = nn.Conv2d(perc_chn, hidden_n, 1)
        self.w2 = nn.Conv2d(hidden_n, self.public, 1, bias=False)  
        self.w2.weight.data.zero_()

        # Learnable PDE Parameters (Raw latent representations)
        self.mu = nn.Parameter(torch.tensor(0.1)) # Decay rate of a, b
        self.omega = nn.Parameter(torch.tensor(0.4)) # Angular drift frequency
        self.beta_r = nn.Parameter(torch.tensor(0.3)) # Cubic amplitude saturation strength
        self.beta_i = nn.Parameter(torch.tensor(0.1))  #Shear / detuning
        self.K = nn.Parameter(torch.tensor(0.25)) # Latent Activator spatial coupling 
        self.raw_beta_d = nn.Parameter(torch.tensor(-1.0)) # Latent decay rate of d (softplus will be apply)
        self.raw_kappa = nn.Parameter(torch.tensor(-1.5)) # Latent d-field diffusion strength
        self.dt = 0.1

        # Inputs for slow RA perception
        self.slow_input_net = nn.Conv2d(9, 3, kernel_size=1)
        
        # Modulation channels
        self.mod_output_net = nn.Conv2d(3, 3, kernel_size=1)
        nn.init.zeros_(self.mod_output_net.weight)
        nn.init.zeros_(self.mod_output_net.bias)

        # FiLM Modulation Layers
        self.film_gamma = nn.Conv2d(3, hidden_n, 1)
        self.film_beta  = nn.Conv2d(3, hidden_n, 1)
        
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.normal_(self.film_beta.weight, std=0.01)
        nn.init.zeros_(self.film_beta.bias)

    def get_alive_mask(self, x):
        alpha = x[:, 3:4, :, :] 
        padded_alpha = F.pad(alpha, pad=[1, 1, 1, 1], mode="circular")
        return F.max_pool2d(padded_alpha, 3, stride=1, padding=0) > 0.1

    def forward(self, x, update_rate=0.5, step=0, k=4):
        # 1. Split state channels
        prefix = x[:, :16]    # RGBA + Hidden (Public)
        a = x[:, 16:17]       # Oscillator a
        b = x[:, 17:18]       # Oscillator b
        d = x[:, 18:19]       # Diffusion field d
        
        # Consistent neighborhood alive mask
        live_mask = self.get_alive_mask(x).to(x.dtype)
        
        # 2. Slow Ring Attractor PDE Updates (Every k steps)
        if step % k == 0:
            Q = slow_perception(x[:, :4], x[:, 4:16])
            I_signals = self.slow_input_net(Q)
            Ia, Ib, Id = I_signals[:, 0:1], I_signals[:, 1:2], I_signals[:, 2:3]

            Ia = Ia * live_mask
            Ib = Ib * live_mask
            Id = Id * live_mask
            
            beta_phys  = F.softplus(self.raw_beta_d)  + 1e-4
            kappa_phys = F.softplus(self.raw_kappa) + 1e-4
            mu_phys = F.softplus(self.mu) + 1e-4
            betar_phys = F.softplus(self.beta_r) + 1e-4
            K_phys = F.softplus(self.K) + 1e-4
            
            new_a, new_b, new_d = discrete_update(
                a, b, d, mu_phys, self.omega, betar_phys, self.beta_i, beta_phys, kappa_phys, 
                K_phys, Ia, Ib, Id, dt=self.dt, live_mask=live_mask)
            #new_a, new_b = consensus_update(new_a, new_b, dt=self.dt, mode='local')
            
            a = new_a * live_mask + a * (1.0 - live_mask)
            b = new_b * live_mask + b * (1.0 - live_mask)
            d = new_d * live_mask + d * (1.0 - live_mask)
            
        # 3. Compute FiLM Modulation Signals from RA states
        ra_stack = torch.cat([a, b, d], dim=1)
        m = torch.sigmoid(self.mod_output_net(ra_stack)) 
        
        m_g = m[:, 0:1]  # growth gate
        m_r = m[:, 1:2]  # regeneration gate
        m_s = m[:, 2:3]  # maintenance gate

        # Standard unconstrained FiLM scaling
        film_gamma_val = 1.0 + self.film_gamma(m)
        film_beta_val  = self.film_beta(m)

        # 4. Fast NCA Processing with FiLM Modulation
        pre_life_mask = self.get_alive_mask(prefix).to(x.dtype)
        fast_input = reduced_perception(prefix, 0)
        
        z = self.w1(fast_input)
        z_prime = film_gamma_val * z + film_beta_val        
        y = self.w2(F.relu(z_prime))
        
        # Correct stochastic update mask
        b_sz, c_sz, h, w = y.shape
        update_mask = (torch.rand(b_sz, 1, h, w, device=x.device) < update_rate).to(x.dtype)
        delta = y * update_mask * pre_life_mask
        
        new_public = prefix + delta
        post_life_mask = self.get_alive_mask(new_public).to(x.dtype)
        new_public = new_public * post_life_mask

        # 5. Re-assemble state tensor
        x_final = torch.cat([new_public, a, b, d, m_g, m_r, m_s], dim=1)

        amplitude, phase = ring_attractor_phases(a, b)
        return x_final, amplitude, phase, film_gamma_val, film_beta_val




class NCA_onlyRA(torch.nn.Module):
    def __init__(self, chn=19, hidden_n=96, recurrent =3):
        super().__init__()
        self.chn = chn
        self.public = chn - recurrent   # NCA fast only read RGBA+hidden to create the perception vector 

        
        dummy = torch.zeros([1, self.public, 8, 8], device="cuda:0")
        perc_chn = reduced_perception(dummy, 0).shape[1]
        
        
        # The MLP works as same as the baseline NCA, the RA dynamics are only injected at the end, they do not participate on the MLP
        self.w1 = torch.nn.Conv2d(perc_chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, self.public, 1, bias=False)  
        self.w2.weight.data.zero_()
        
        
        #Parameter of the RA 
        self.alpha = torch.nn.Parameter(torch.tensor(0.1)) # Decay rate of the activator/phase
        self.beta  = torch.nn.Parameter(torch.tensor(-1.0)) # Decay rate of the inhibitor/injury (sofplus will be applied)
        self.omega = torch.nn.Parameter(torch.tensor(0.0)) # Angular drift
        self.K     = torch.nn.Parameter(torch.tensor(-2.0)) # Spatial coupling between activator and inhibitor (sofplus will be applied)
        self.kappa = torch.nn.Parameter(torch.tensor(-1.0)) # Diffusion strength (sofplus will be applied)
        self.dt    = 0.1

        # Inputs for the slow perception of the RA 
        # Q -> Ia, Ib, Id
        self.slow_input_net = torch.nn.Conv2d(5, 3, kernel_size=1)
        # With FiLM modulation we take the a,b,d states to the mod channels 
        # a,b,d -> m_g, m_s, m_r
        #FiLM modulation
        self.mod_gamma = torch.nn.Conv2d(3, hidden_n, 1)
        self.mod_beta  = torch.nn.Conv2d(3, hidden_n, 1)
        
        # Initialization on zeros as the NCA architecture does 
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


        # Slow RA updates
        if step % k == 0 : # Update the RA every k steps (including the first step)
            Q = slow_perception(x[:, :4], x[:, 4:16]) 
            I_signals = torch.tanh(self.slow_input_net(Q)) #Constraint of the signal to [-1,1]
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
        pre_life_mask = self.get_alive_mask(x)
        fast_input = reduced_perception(x[:, :self.public], 0) # We only use the RGBA + hidden for the fast perception
        h = self.w1(fast_input)
        h = gamma * h + beta
        y = self.w2(torch.relu(h)) 
        
        # Masks
        b_sz, c_sz, h, w = y.shape
        update_mask = (torch.rand(b_sz, 1, h, w, device=x.device) + update_rate).floor()



        #delta update 
        delta = y * update_mask * pre_life_mask.to(y.dtype)
        post_life_mask = self.get_alive_mask(x)

        #  Update of the new public channels (prefix)
        new_public =  (prefix + delta) * post_life_mask
        # We concatenate all parts to create x_final without ever modifying the input x
        x_final = torch.cat([
            new_public, # 0:16
            a,          # 16
            b,          # 17
            d,          # 18
        ], dim=1)

        amplitude, phase = ring_attractor_phases(a, b)
        return x_final, amplitude, phase




class NCA_onlymod(torch.nn.Module):
    def __init__(self, public=16, m_dim=3, hidden_n=64, m_mode='fixed'):
        super().__init__()
        self.public = public
        self.m_dim = m_dim
        self.m_mode = m_mode  # It can be fixed or with a small feedforward network  

        dummy = torch.zeros([1, self.public, 8, 8])
        perc_chn = reduced_perception(dummy, 0).shape[1]

        self.w1 = torch.nn.Conv2d(perc_chn, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, self.public, 1, bias=False)
        self.w2.weight.data.zero_()

        # FiLM conditioned on the modulation channels 
        self.film_gamma = torch.nn.Conv2d(m_dim, hidden_n, 1)
        self.film_beta  = torch.nn.Conv2d(m_dim, hidden_n, 1)
        torch.nn.init.zeros_(self.film_gamma.weight)
        torch.nn.init.zeros_(self.film_gamma.bias)
        torch.nn.init.normal_(self.film_beta.weight, std=0.01)
        torch.nn.init.zeros_(self.film_beta.bias)

        if m_mode == 'feedforward':
            self.m_feedforward = torch.nn.Conv2d(public, m_dim, 1)

    def get_alive_mask(self,x):
        alpha = x[:, 3:4, :, :] 
        padded_alpha = torch.nn.functional.pad(alpha, pad=[1, 1, 1, 1], mode="circular")
        return torch.nn.functional.max_pool2d(padded_alpha, 3, stride=1, padding=0) > 0.1

    def forward(self, x, update_rate=0.5):
        prefix = x[:, :self.public].clone()

        if self.m_mode == 'fixed':
            m = x[:, self.public:self.public + self.m_dim].clone()  # fix never updated
        else: 
            m = torch.sigmoid(self.m_feedforward(x[:, :self.public]))  # recomputed fresh each step depending on the public channels 

        pre_life_mask = self.get_alive_mask(x)
        fast_input = reduced_perception(x[:, :self.public], 0)
        z = self.w1(fast_input)    # Firs MLP channel for the FiLM modulation
        # Unbounded FiLM modulation
        gamma = 1.0 + self.film_gamma(m)
        beta  = self.film_beta(m)
        z_prime = gamma * z + beta    
        y = self.w2(torch.relu(z_prime))

        # Update Mask considering an lower bound for the random values 
        b_sz, c_sz, h, w = y.shape
        update_mask = (torch.rand(b_sz, 1, h, w, device=x.device) < update_rate).to(x.dtype)

        
        delta = y * update_mask * pre_life_mask
        new_public = (prefix + delta) 
        post_life_mask = self.get_alive_mask(new_public).to(x.dtype)
        new_public = new_public * post_life_mask

        x_final = torch.cat([new_public, m], dim=1)
        return x_final




