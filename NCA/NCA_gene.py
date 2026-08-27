from multiprocessing import dummy
from sys import prefix

import torch
import torch.nn as nn
import torch.nn.functional as F

ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda:0")
sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device="cuda:0")
sobel_y = sobel_x.T
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
DEVICE = "cuda:0"

lap_slow = torch.tensor([[0., 1., 0.],
                     [1., -4., 1.],
                     [0., 1., 0.]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)

gaus_slow = torch.tensor([[1., 2., 1.],
                      [2., 4., 2.],
                      [1., 2., 1.]], dtype=torch.float32, device=DEVICE) / 16.0
gaus_slow = gaus.view(1, 1, 3, 3)

sobel_x_slow = torch.tensor([[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)

sobel_y_slow = torch.tensor([[-1., -2., -1.],
                         [ 0.,  0.,  0.],
                         [ 1.,  2.,  1.]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)


def ring_attractor_phases(a, b):
    local_amplitude = torch.sqrt(a**2 + b**2 + 1e-6)
    local_angle = torch.atan2(b, a)
    return local_amplitude, local_angle


# We apply the live_mask on the discrete update to no have a growth of a,b,d on the background
def discrete_update(a, b, d, mu, omega, g0, c, beta_d, kappa, Kr, Ki,
                     I_a, I_b, I_d, dt, live_mask):
    a_padded = torch.nn.functional.pad(a, [1, 1, 1, 1], mode='circular')
    diff_a = torch.nn.functional.conv2d(a_padded, lap_slow, padding=0)
    b_padded = torch.nn.functional.pad(b, [1, 1, 1, 1], mode='circular')
    diff_b = torch.nn.functional.conv2d(b_padded, lap_slow, padding=0)

    r_sq = a**2 + b**2
    cubic_a = -r_sq * (g0 * a - c * b)
    cubic_b = -r_sq * (g0 * b + c * a)

    # \dot{z}(x,t)= (\mu+i\omega_0)z+(D_r+iD_i)\nabla^2z-(g_0+ic)|z|^2z +I^z
    new_a = a + dt * (mu * a - omega * b + cubic_a + Kr * diff_a - Ki * diff_b + I_a)
    new_b = b + dt * (mu * b + omega * a + cubic_b + Kr * diff_b + Ki * diff_a + I_b)

    d_padded = torch.nn.functional.pad(d, [1, 1, 1, 1], mode='circular')
    diff_d = torch.nn.functional.conv2d(d_padded, lap_slow, padding=0)
    new_d = d + dt * (-beta_d * d + kappa * diff_d + I_d)

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
    h_layers = hidden[:, 0:2, :, :]  # arbitrary public hidden channels used as slow-PDE input

    # 1. Padding
    alpha_padded = torch.nn.functional.pad(alpha, [1, 1, 1, 1], mode='circular')

    # 2. Convolutions with external filters
    lap_alpha = F.conv2d(alpha_padded, lap_slow)
    lap_inward = -lap_alpha

    # 3. Gradients in x and y to have orientation
    smooth_alpha = F.conv2d(alpha_padded, gaus_slow)
    #grad_x = F.conv2d(alpha_padded, sobel_x_slow)
    #grad_y = F.conv2d(alpha_padded, sobel_y_slow)

    # 4. Morphological Boundary
    eroded = -F.max_pool2d(-alpha_padded, kernel_size=3, stride=1, padding=0)
    dilated = F.max_pool2d(alpha_padded, kernel_size=3, stride=1, padding=0)
    morph_edge = dilated - eroded

    Q = torch.cat([alpha, lap_inward, smooth_alpha, eroded, morph_edge, h_layers], dim=1)
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
        self.mu = nn.Parameter(torch.tensor(0.26)) # Decay rate of a, b
        self.omega = nn.Parameter(torch.tensor(0.3)) # Angular drift frequency
        self.g0 = nn.Parameter(torch.tensor(0.16)) # Cubic amplitude saturation strength
        self.c = nn.Parameter(torch.tensor(0.0))  #Shear / detuning
        self.Kr = nn.Parameter(torch.tensor(1.2)) # Latent Activator spatial coupling (amplitude)
        self.Ki = nn.Parameter(torch.tensor(0.3)) # Latent Activator spatial coupling (phase)
        self.raw_beta_d = nn.Parameter(torch.tensor(-1.0)) # Latent decay rate of d (softplus will be apply)
        self.raw_kappa = nn.Parameter(torch.tensor(1.0)) # Latent d-field diffusion strength
        self.dt = 0.1

        # Inputs for slow RA perception
        self.slow_input_net = nn.Conv2d(7, 3, kernel_size=1)
        
        # Modulation channels: amplitude, regeneration and competence 
        self.amp_to_gate  = nn.Conv2d(1, 1, kernel_size=1) 
        self.reg_to_gate = nn.Conv2d(4, 1, kernel_size=1)  
        self.d_to_gate    = nn.Conv2d(1, 1, kernel_size=1) 
        

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
        m = x[:, 19:22]
        
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
            g0_phys = F.softplus(self.g0) + 1e-4
            K_r_phys = F.softplus(self.Kr) + 1e-4
            
            new_a, new_b, new_d = discrete_update(
                a, b, d, mu_phys, self.omega, g0_phys, self.c, beta_phys, kappa_phys, 
                K_r_phys, self.Ki , Ia, Ib, Id, dt=self.dt, live_mask=live_mask)
            #new_a, new_b = consensus_update(new_a, new_b, dt=self.dt, mode='local')
            
            a = new_a * live_mask 
            b = new_b * live_mask 
            d = new_d * live_mask 

        # Computation of the inputs to the modulation channels 
        r_sq = a**2 + b**2
        a_pad = F.pad(a, [1,1,1,1], mode='circular')
        b_pad = F.pad(b, [1,1,1,1], mode='circular')
        
        ax = F.conv2d(a_pad, sobel_x_slow)
        ay = F.conv2d(a_pad, sobel_y_slow)
        bx = F.conv2d(b_pad, sobel_x_slow)
        by = F.conv2d(b_pad, sobel_y_slow)

        r_sq_safe = torch.clamp(r_sq, min=0.01)

        r_sq_pad = F.pad(r_sq_safe, [1,1,1,1], mode='circular')
        d_pad    = F.pad(d,         [1,1,1,1], mode='circular')
        r_sq_lap = F.conv2d(r_sq_pad, lap_slow)
        d_lap    = F.conv2d(d_pad,    lap_slow)

        reg_input = torch.cat([r_sq_lap, d_lap, r_sq_safe, d], dim=1)

        # 3. Compute FiLM Modulation Signals from RA states
        m_amp      = torch.sigmoid(self.amp_to_gate(r_sq))
        m_regeneration = torch.sigmoid(self.reg_to_gate(reg_input))
        m_d        = torch.sigmoid(self.d_to_gate(d))

        # Standard unconstrained FiLM scaling
        m = torch.cat([m_amp, m_regeneration , m_d], dim=1)
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
        x_final = torch.cat([new_public, a, b, d, m_amp, m_regeneration, m_d], dim=1)

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
        return x_final, gamma, beta




