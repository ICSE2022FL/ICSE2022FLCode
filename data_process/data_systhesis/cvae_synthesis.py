import math

import torch.utils.data as Data
import numpy as np
import argparse

import pandas as pd
from data_process.ProcessedData import ProcessedData
from data_process.data_systhesis.CVAE_model import *


class CVAESynthesisData(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = raw_data.rest_columns

    def process(self):
        if len(self.label_df) < 2:
            return

        equal_zero_index = (self.label_df != 1).values
        equal_one_index = ~equal_zero_index

        pass_feature = np.array(self.feature_df[equal_zero_index])
        fail_feature = np.array(self.feature_df[equal_one_index])

        diff_num = len(pass_feature) - len(fail_feature)

        if diff_num < 1:
            return


        min_batch = 40
        batch_size = min_batch if len(self.label_df) >= min_batch else len(self.label_df)
        torch_dataset = Data.TensorDataset(torch.tensor(self.feature_df.values, dtype=torch.float32),
                                           torch.tensor(self.label_df.values, dtype=torch.int64))
        loader = Data.DataLoader(dataset=torch_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 )
        input_dimension = len(self.feature_df.values[0])
        hidden_dimension = math.floor(math.sqrt(input_dimension))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        parser = argparse.ArgumentParser()
        parser.add_argument("--encoder_layer_sizes",
                            type=list,
                            default=[input_dimension, hidden_dimension])
        parser.add_argument("--decoder_layer_sizes",
                            type=list,
                            default=[hidden_dimension, input_dimension])
        # the latent_size should be changed for different programs
        parser.add_argument("--latent_size", type=int, default=5)
        parser.add_argument("--conditional", type=bool, default=True)
        parser.add_argument("--lr", type=float, default=0.005)
        args = parser.parse_args()
        cvae = CVAE(encoder_layer_sizes=args.encoder_layer_sizes,
                    latent_size=args.latent_size,
                    decoder_layer_sizes=args.decoder_layer_sizes,
                    conditional=args.conditional,
                    num_labels=2).to(device)
        optimizer = torch.optim.Adam(cvae.parameters(), lr=args.lr)
        EPOCH = 1000
        for epoch in range(EPOCH):
            cvae.train()
            train_loss = 0
            for step, (x, y) in enumerate(loader):
                x = x.unsqueeze(0).unsqueeze(0).to(device)
                y = y.unsqueeze(0).unsqueeze(0).to(device)
                recon_x, mu, logvar, z = cvae(x, y)
                loss = loss_fn(recon_x, x, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            if epoch % 100 == 0:
                print('====>CVAE training... Epoch: {} Average loss: {:.4f}'.format(epoch,
                                                                                   train_loss / len(loader.dataset)))

        with torch.no_grad():
            c = torch.ones(diff_num).long().unsqueeze(1).to(device)
            z = torch.randn([c.size(0), args.latent_size]).to(device)
            x = cvae.inference(z, c=c).to("cpu").numpy()
        features_np = np.array(self.feature_df)
        compose_feature = np.vstack((features_np, x))

        label_np = np.array(self.label_df)
        gen_label = np.ones(diff_num).reshape((-1, 1))
        compose_label = np.vstack((label_np.reshape(-1, 1), gen_label))

        self.label_df = pd.DataFrame(compose_label, columns=['error'], dtype=float)
        self.feature_df = pd.DataFrame(compose_feature, columns=self.feature_df.columns, dtype=float)

        self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)
