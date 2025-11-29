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


class LogScale:
    """
    x -> log(x) + eps
    """

    @staticmethod
    def forward(x, indices, eps=1e-6):
        return x[..., indices].add(eps).log()

    @staticmethod
    def reverse(x, indices):
        return x[..., indices].exp()


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

        # process reco level
        batch.x[:, 1] += torch.rand_like(batch.x[:, 1]) - 0.5
        batch.x[:, :4] = batch.x[:, :4].add(self.eps).log()
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)

        # process part level
        batch.z[:, 1] += torch.rand_like(batch.z[:, 1]) - 0.5
        batch.z[:, :4] = batch.z[:, :4].add(self.eps).log()
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)

        return batch

    def reverse(self, batch):

        # process reco level
        batch.x = ShiftAndScale.reverse(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.x[:, :4] = batch.x[:, :4].exp().sub(self.eps)
        batch.x[:, 1] = batch.x[:, 1].round()

        # process part level
        batch.z = ShiftAndScale.reverse(batch.z, shift=self.shift_z, scale=self.scale_z)
        batch.z[:, :4] = batch.z[:, :4].exp().sub(self.eps)
        batch.z[:, 1] = batch.z[:, 1].round()

        return batch


# TODO: split into Transformx and Transformz
@tensorclass
class OmniFoldParticleTransform:
    """
    TODO
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-8

    def forward(self, batch):

        # process reco level
        batch.x[..., 0] = batch.x[..., 0].add(self.eps).log()
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)

        # process part level
        batch.z[..., 0] = batch.z[..., 0].add(self.eps).log()
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)

        return batch

    def reverse(self, batch):

        # process reco level
        batch.x = ShiftAndScale.reverse(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.x[:, :4] = batch.x[:, :4].exp().sub(self.eps)
        batch.x[:, 1] = batch.x[:, 1].round()

        # process part level
        batch.z = ShiftAndScale.reverse(batch.z, shift=self.shift_z, scale=self.scale_z)
        batch.z[:, :4] = batch.z[:, :4].exp().sub(self.eps)
        batch.z[:, 1] = batch.z[:, 1].round()

        return batch


@tensorclass
class ttbarTransform:
    """
    TODO
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-3

    def forward(self, batch):

        # process reco level
        Ms_x = [
            compute_invariant_mass(batch.x, ps).unsqueeze(1)
            for ps in ([0, 1], [0, 2], [1, 2], [0, 1, 2])
        ]
        batch.x = batch.x.flatten(-2, -1)
        batch.x = torch.cat([batch.x, *Ms_x], dim=1)
        batch.x[..., [0, 1, 4, 5, 8, 9]] = (
            batch.x[..., [0, 1, 4, 5, 8, 9]].add(self.eps).log()
        )
        # batch.x[..., [12, 13, 14]] -= 80.3
        # batch.x[..., 15] -= 172.5
        # batch.x[..., [12, 13, 14, 15]] = batch.x[..., [12, 13, 14, 15]].arcsinh()
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)

        # process part level
        Ms_z = [
            compute_invariant_mass(batch.z, ps).unsqueeze(1)
            for ps in ([0, 1], [0, 2], [1, 2], [0, 1, 2])
        ]
        batch.z = batch.z.flatten(-2, -1)
        batch.z = torch.cat([batch.z, *Ms_z], dim=1)
        batch.z[..., [0, 1, 4, 5, 8, 9]] = (
            batch.z[..., [0, 1, 4, 5, 8, 9]].add(self.eps).log()
        )
        # batch.z[..., [12, 13, 14]] -= 80.3
        # batch.z[..., 15] -= 172.5
        # batch.z[..., [12, 13, 14, 15]] = batch.z[..., [12, 13, 14, 15]].arcsinh()
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)

        return batch

    def reverse(self, batch):

        # process reco level
        batch.x = batch.x.unflatten(-1, (3, 4))
        batch.x = ShiftAndScale.reverse(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.x[..., :2] = batch.x[..., :2].exp().sub(self.eps)

        # process part level
        batch.z = batch.z.unflatten(-1, (3, 4))
        batch.z = ShiftAndScale.reverse(batch.z, shift=self.shift_z, scale=self.scale_z)
        batch.z[..., :2] = batch.z[..., :2].exp().sub(self.eps)

        return batch


def compute_invariant_mass(p, particles) -> torch.Tensor:

    px_sum = 0
    py_sum = 0
    pz_sum = 0
    e_sum = 0
    for particle in particles:

        m = p[..., particle, 0]
        pT = p[..., particle, 1]
        eta = p[..., particle, 2]
        phi = p[..., particle, 3]

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        e = torch.sqrt(m**2 + px**2 + py**2 + pz**2)

        px_sum += px
        py_sum += py
        pz_sum += pz
        e_sum += e

    m = torch.sqrt(
        torch.clamp((e_sum) ** 2 - (px_sum) ** 2 - (py_sum) ** 2 - (pz_sum) ** 2, min=0)
    )
    return m
