import torch
import torch.nn as nn

# Cribbed from: 
# https://github.com/JindongJiang/RedNet/blob/master/RedNet_model.py
# https://github.com/ajgallego/document-image-binarization/blob/master/utilModelREDNet.py
class RedNet(nn.Module):
  def __init__(self, nb_filters=32, k_size=5, stride=1):
    super(RedNet, self).__init__()
    self.nb_layers = 5

    self.initialConv = nn.ConvTranspose2d(
      in_channels=3, 
      out_channels=nb_filters, 
      kernel_size=k_size, 
      stride=stride
    )

    encoderLayers = [None] * self.nb_layers
    decoderLayers = [None] * self.nb_layers

    for i in range(self.nb_layers):
      encoderLayers[i] = nn.Sequential(
        nn.Conv2d(
          in_channels=nb_filters,
          out_channels=nb_filters, 
          kernel_size=k_size, 
          stride=stride
        ),
        nn.BatchNorm2d(nb_filters),
        nn.ReLU()
      )

    for i in range(self.nb_layers):
      decoderLayers[i] = nn.Sequential(
        nn.ConvTranspose2d(
          in_channels=nb_filters, 
          out_channels=nb_filters,
          kernel_size=k_size,
          stride=stride
        ),
        nn.BatchNorm2d(nb_filters),
        nn.ReLU()
      )

    # Register for optimization
    self.encoderLayers = nn.Sequential(*encoderLayers)
    self.decoderLayers = nn.Sequential(*decoderLayers)

    self.finalConv = nn.Sequential(
      nn.Conv2d(nb_filters, 1, kernel_size=k_size, stride=stride),
      nn.Sigmoid()
    )


  def forward(self, rgb):
    x = self.initialConv(rgb)

    agant = [None] * self.nb_layers    
    for i, encoder in enumerate(self.encoderLayers):
      # print("encode", i, [p.is_cuda for p in encoder.parameters()])
      x = encoder(x)
      # print("encode done", x.shape)
      agant[i] = x

    for i, decoder in enumerate(self.decoderLayers):
      reachback = self.nb_layers - i - 1
      # print("decode", i, reachback)
      x = x + agant[reachback]
      x = decoder(x)
      # print("encode done", x.shape)

    x = self.finalConv(x)
    return x

def rednet():
  return RedNet()
