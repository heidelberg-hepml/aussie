import torch
from tensordict import tensorclass

class ShiftAndScale:
    """
    x -> (x - shift) / scale
    """

    @staticmethod
    def forward(x, shift=0, scale=1):
        return x.sub(shift).div(scale)

    @staticmethod
    def reverse(x, shift=0, scale=1):
        return x.multiply(scale).add(shift)

# TODO: split into Transformx and Transformz
@tensorclass
class OmniFoldTransform:
    """
    TODO
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-3

    def forward(self, batch):
        
        if 'x_sim' in batch.keys():        
            batch.x_sim[:, :-1] = batch.x_sim[:, :-1].add(self.eps).log()
            batch.x_sim = ShiftAndScale.forward(batch.x_sim, shift=self.shift_x, scale=self.scale_x)
        
        if 'x_dat' in batch.keys():
            batch.x_dat[:, :-1] = batch.x_dat[:, :-1].add(self.eps).log()
            batch.x_dat = ShiftAndScale.forward(batch.x_dat, shift=self.shift_x, scale=self.scale_x)

        if 'z_sim' in batch.keys():        
            batch.z_sim[:, :-1] = batch.z_sim[:, :-1].add(self.eps).log()
            batch.z_sim = ShiftAndScale.forward(batch.z_sim, shift=self.shift_z, scale=self.scale_z)
        
        if 'z_dat' in batch.keys():
            batch.z_dat[:, :-1] = batch.z_dat[:, :-1].add(self.eps).log()
            batch.z_dat = ShiftAndScale.forward(batch.z_dat, shift=self.shift_z, scale=self.scale_z)            

        return batch

    def reverse(self, batch):

        if 'x_sim' in batch.keys():        
            batch.x_sim = ShiftAndScale.reverse(batch.x_sim, shift=self.shift_x, scale=self.scale_x)
            batch.x_sim[:, :-1] = batch.x_sim[:, :-1].exp().sub(self.eps)
        
        if 'x_dat' in batch.keys():
            batch.x_dat = ShiftAndScale.reverse(batch.x_dat, shift=self.shift_x, scale=self.scale_x)
            batch.x_dat[:, :-1] = batch.x_dat[:, :-1].exp().sub(self.eps)

        if 'z_sim' in batch.keys():        
            batch.z_sim = ShiftAndScale.reverse(batch.z_sim, shift=self.shift_z, scale=self.scale_z)
            batch.z_sim[:, :-1] = batch.z_sim[:, :-1].exp().sub(self.eps)
        
        if 'z_dat' in batch.keys():
            batch.z_dat = ShiftAndScale.reverse(batch.z_dat, shift=self.shift_z, scale=self.scale_z)            
            batch.z_dat[:, :-1] = batch.z_dat[:, :-1].exp().sub(self.eps)

        return batch
