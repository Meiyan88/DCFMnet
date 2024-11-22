import torch
from torch import nn
from fdm import generate_model,fdm

class fmm(nn.Module):
    def __init__(self, args):
        super(fmm, self).__init__()

        self.args = args

        self.mri_encoder1 = generate_model(10)
        self.mri_encoder2 = generate_model(10)

        self.VectorQuantizer_mri_com_0 = VectorQuantizer(self.args, 128, 100, 0.25)
        self.VectorQuantizer_pet_com_0 = VectorQuantizer(self.args, 128, 100, 0.25)
        self.VectorQuantizer_mri_spe_0 = VectorQuantizer(self.args, 128, 200, 0.25)
        self.VectorQuantizer_pet_spe_0 = VectorQuantizer(self.args, 128, 200, 0.25)

        self.rec_mri_decoder = fdm.Rec_Decoder(conf=self.args)
        self.rec_pet_decoder = fdm.Rec_Decoder(conf=self.args)

        self.spe = Class_linear(500)
        self.com = Class_linear(400)

    def forward(self, mri,com_m,com_p,spe_m,spe_p):


        logist_mri1, x_m1 = self.mri_encoder1(mri)
        logist_mri2, x_m2 = self.mri_encoder2(mri)

        loss_pet_com_0, min_encodings_pet_com, pet_com_0, pet_com_dic_0,indices_pet_com = self.VectorQuantizer_pet_com_0(com_p)
        loss_mri_com_0, min_encodings_mri_com, mri_com_0, mri_com_dic_0, indices_mri_com = self.VectorQuantizer_mri_com_0(com_m)
        loss_mri_spe_0, min_encodings_0, mri_spe_0, mri_spe_dic_0,indices_mri_spe = self.VectorQuantizer_mri_spe_0(spe_m)
        loss_pet_spe_0, min_encodings_0_real, pet_spe_0, pet_spe_dic_0,indices_pet_spe = self.VectorQuantizer_pet_spe_0(spe_p)

        x_m_1 = torch.cat((pet_com_0, mri_spe_0), dim=1)
        x_p_1 = torch.cat((mri_com_0, pet_spe_0), dim=1)

        x_m_2 = torch.cat((mri_com_0, mri_spe_0), dim=1)
        x_p_2 = torch.cat((pet_com_0, pet_spe_0), dim=1)

        rec_mri_1 = self.rec_mri_decoder(x_m_1).to(self.args.device[0])
        rec_pet_1 = self.rec_pet_decoder(x_p_1).to(self.args.device[0])

        rec_mri_2 = self.rec_mri_decoder(x_m_2).to(self.args.device[0])
        rec_pet_2 = self.rec_pet_decoder(x_p_2).to(self.args.device[0])


        logit1_spe = self.spe(torch.cat((x_m1[0, :], pet_spe_0[0, :].to(self.args.device[0])), dim=0))
        logit2_spe = self.spe(torch.cat((x_m1[1, :], pet_spe_0[1, :].to(self.args.device[0])), dim=0))
        logit3_spe = self.spe(torch.cat((x_m1[0, :], pet_spe_0[1, :].to(self.args.device[0])), dim=0))
        logit4_spe = self.spe(torch.cat((x_m1[1, :], pet_spe_0[0, :].to(self.args.device[0])), dim=0))
        logits_spe = torch.cat((logit1_spe, logit2_spe, logit3_spe, logit4_spe), dim=0).unsqueeze(dim=1)

        logit1_com = self.com(torch.cat((x_m2[0, :], pet_com_0[0, :].to(self.args.device[0])), dim=0))
        logit2_com = self.com(torch.cat((x_m2[1, :], pet_com_0[1, :].to(self.args.device[0])), dim=0))
        logit3_com = self.com(torch.cat((x_m2[0, :], pet_com_0[1, :].to(self.args.device[0])), dim=0))
        logit4_com = self.com(torch.cat((x_m2[1, :], pet_com_0[0, :].to(self.args.device[0])), dim=0))
        logits_com = torch.cat((logit1_com, logit2_com, logit3_com, logit4_com), dim=0).unsqueeze(dim=1)


        return rec_mri_1,rec_pet_1,rec_mri_2,rec_pet_2,logits_spe,logits_com


class VectorQuantizer(nn.Module):

    def __init__(self, args,n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.args = args
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):

        z_flattened = z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(self.args.device[0])
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        loss = torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()

        return loss, min_encodings,z_q,self.embedding.weight,min_encoding_indices

class Class_linear(nn.Module):
    def __init__(self, dim=300, conf=None):
        super(Class_linear, self).__init__()

        self.conf = conf
        self.finally_layer = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        out = self.finally_layer(x)

        return torch.sigmoid(out)