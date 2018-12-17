import torch

# micro_fm
def loss_function():
  def ll(y_pred, y_true):
    beta = 1.0
    beta2 = beta**2.0
    top = torch.sum(y_true * y_pred)
    bot = beta2 * torch.sum(y_true) + torch.sum(y_pred)
    return -(1.0 + beta2) * top / bot
  return ll
