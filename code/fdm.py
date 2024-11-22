import torch
import torch.nn as nn
from decoder import Rec_Decoder
from encoder import generate_model
class fdm(nn.Module):
    def __init__(self,args):
        super(fdm,self ).__init__()

        self.args = args

        self.mri_encoder = generate_model(10)
        self.pet_encoder = generate_model(10)

        self.rec_mri_decoder = Rec_Decoder(conf=self.args)
        self.rec_pet_decoder = Rec_Decoder(conf=self.args)
    def forward(self, mri, pet):
        com_m, spe_m= self.mri_encoder(mri)
        com_p, spe_p = self.pet_encoder(pet)

        x_m1 = torch.cat((com_p, spe_m), dim=1)
        x_p1 = torch.cat((com_m, spe_p), dim=1)
        x_m2 = torch.cat((com_m, spe_m), dim=1)
        x_p2 = torch.cat((com_p, spe_p), dim=1)

        rec_mri_1 = self.rec_mri_decoder(x_m1)
        rec_pet_1 = self.rec_pet_decoder(x_p1)
        rec_mri_2 = self.rec_mri_decoder(x_m2)
        rec_pet_2 = self.rec_pet_decoder(x_p2)

        return com_m,com_p,spe_m,spe_p,rec_mri_1,rec_pet_1,rec_mri_2,rec_pet_2





