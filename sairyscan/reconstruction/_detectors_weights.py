import math
import torch


class SAiryscanWeights:
    def __init__(self, method='mean'):
        self.method = method

    def __call__(self):
        if self.method == 'mean':
            weights = torch.ones((32,))
            return weights / torch.sum(weights)
        elif self.method == 'ring':
            return self.ring_distance(3.0)
        elif self.method == 'ring_inv':
            return self.ring_inv_distance(3.0)
        elif self.method == 'd2c':
            d2c = self.distance_to_center(1)
            return d2c / torch.sum(d2c)
        elif self.method == 'id2c':
            id2c = 3.0 - self.distance_to_center(1)
            return id2c / torch.sum(id2c)
        elif self.method == 'exp_d2c':
            return self.exp_distance_to_center(3.0)
        elif self.method == 'exp_d2c_inv':
            return self.exp_inv_distance_to_center(3.0)

    @staticmethod
    def ring_distance(tau):
        weights = torch.ones((32,))
        for i in range(1, 7):
            weights[i] = math.exp(1/tau)
        for i in range(7, 19):
            weights[i] = math.exp(2 / tau)
        for i in range(19, 32):
            weights[i] = math.exp(3 / tau)
        return weights/torch.sum(weights)

    @staticmethod
    def ring_inv_distance(tau):
        weights = torch.ones((32,))
        for i in range(1, 7):
            weights[i] = math.exp(-1 / tau)
        for i in range(7, 19):
            weights[i] = math.exp(-2 / tau)
        for i in range(19, 32):
            weights[i] = math.exp(-3 / tau)
        return weights/torch.sum(weights)

    @staticmethod
    def exp_distance_to_center(tau):
        weights = SAiryscanWeights.distance_to_center(1)
        for i in range(32):
            weights[i] = math.exp(weights[i] / tau)
        return weights/torch.sum(weights)

    @staticmethod
    def exp_inv_distance_to_center(tau):
        weights = SAiryscanWeights.distance_to_center(1)
        for i in range(32):
            weights[i] = math.exp(-weights[i] / tau)
        return weights/torch.sum(weights)

    @staticmethod
    def distance_to_center(d):
        dist = torch.zeros((32,))
        dist[0] = SAiryscanWeights.norm(0, 0)
        dist[1] = SAiryscanWeights.norm(d, 0.5*d)
        dist[2] = SAiryscanWeights.norm(d, -0.5*d)
        dist[3] = SAiryscanWeights.norm(0, -d)
        dist[4] = SAiryscanWeights.norm(-d, -0.5*d)
        dist[5] = SAiryscanWeights.norm(-d, 0.5*d)
        dist[6] = SAiryscanWeights.norm(0, d)
        dist[7] = SAiryscanWeights.norm(d, 1.5*d)
        dist[8] = SAiryscanWeights.norm(2*d, d)
        dist[9] = SAiryscanWeights.norm(2*d, 0)
        dist[10] = SAiryscanWeights.norm(2*d, -d)
        dist[11] = SAiryscanWeights.norm(d, -1.5*d)
        dist[12] = SAiryscanWeights.norm(0, -2*d)
        dist[13] = SAiryscanWeights.norm(-d, -1.5*d)
        dist[14] = SAiryscanWeights.norm(-2*d, -d)
        dist[15] = SAiryscanWeights.norm(-2*d, 0)
        dist[16] = SAiryscanWeights.norm(-2*d, d)
        dist[17] = SAiryscanWeights.norm(-d, 1.5*d)
        dist[18] = SAiryscanWeights.norm(0, 2*d)
        dist[19] = SAiryscanWeights.norm(d, 2.5*d)
        dist[20] = SAiryscanWeights.norm(2*d, 2*d)
        dist[21] = SAiryscanWeights.norm(3*d, 0.5*d)
        dist[22] = SAiryscanWeights.norm(3*d, -0.5*d)
        dist[23] = SAiryscanWeights.norm(2*d, -2*d)
        dist[24] = SAiryscanWeights.norm(d, -2.5*d)
        dist[25] = SAiryscanWeights.norm(-d, -2.5*d)
        dist[26] = SAiryscanWeights.norm(-2*d, -2*d)
        dist[27] = SAiryscanWeights.norm(-3*d, -0.5*d)
        dist[28] = SAiryscanWeights.norm(-3*d, 0.5*d)
        dist[29] = SAiryscanWeights.norm(-2*d, 2*d)
        dist[30] = SAiryscanWeights.norm(-1*d, 2.5*d)
        dist[31] = SAiryscanWeights.norm(0, 3*d)
        return dist

    @staticmethod
    def norm(a, b):
        return math.sqrt(a*a + b*b)
