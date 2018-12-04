import torch.optim as optim

def optimizer(network):
  return optim.Adam(network.parameters(), lr=0.00001), None
