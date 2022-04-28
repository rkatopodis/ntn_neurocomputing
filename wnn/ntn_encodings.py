import torch


class Thermometer(object):
    def __init__(self, minimum, maximum, resolution):
        self.minimum = minimum
        self.maximum = maximum
        self.resolution = resolution

    def __repr__(self):
        return f"Thermometer(minimum={self.minimum}, maximum={self.maximum}, resolution={self.resolution})"

    def encode(self, X):
        X = torch.as_tensor(X, device='cpu')

        enc = torch.arange(self.resolution).expand(*X.shape, self.resolution)
        threshold = (X.reshape((*X.shape, 1)) - self.minimum)/(self.maximum - self.minimum)*self.resolution

        return (enc < threshold).to(torch.long)


class CircularEncoder(object):
    def __init__(self, minimum, maximum, resolution):
        self.minimum = minimum
        self.maximum = maximum
        self.resolution = resolution

    def __repr__(self):
        return f"CircularEncoder(minimum={self.minimum}, maximum={self.maximum}, resolution={self.resolution}), wrap={self.wrap}"

    def encode(self, X):
        X = torch.atleast_1d(torch.as_tensor(X, device='cpu'))

        pattern = (
            torch.arange(self.resolution).view(1, -1) < self.resolution//2
        ).to(torch.long).repeat(*X.shape, 1)

        shifts = (X - self.minimum)/(self.maximum - self.minimum)*self.resolution
        gather_idx = (
            torch.arange(self.resolution).expand(pattern.shape)
            - shifts.floor().view(*X.shape, 1)
        ) % self.resolution

        return torch.gather(pattern, X.ndim, gather_idx.to(torch.long))
