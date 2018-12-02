import torch.optim as optim

def adam(network):
  return optim.Adam(network.parameters(), lr=0.00001), None
