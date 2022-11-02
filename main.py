import pandas
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet

from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import matplotlib.pyplot as plt

N_GRAPH_PER_EDGE = 64
Epochs = 100
data = MoleculeNet(root=".", name="ESOL")


train_loader = DataLoader(
    data[: int(len(data) * 0.8)], batch_size=N_GRAPH_PER_EDGE, shuffle=True
)

test_loader = DataLoader(
    data[int(len(data) * 0.8) :], batch_size=N_GRAPH_PER_EDGE, shuffle=True
)


class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        torch.manual_seed(41)

        self.g1 = GCNConv(in_channels=input_size, out_channels=hidden_size)
        self.g2 = GCNConv(in_channels=hidden_size, out_channels=hidden_size)
        self.g3 = GCNConv(in_channels=hidden_size, out_channels=hidden_size)
        self.g4 = GCNConv(in_channels=hidden_size, out_channels=hidden_size)
        self.l1 = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, edge_index, batch_index):
        keep_going = torch.tanh(self.g1(x, edge_index))
        keep_going = torch.tanh(self.g2(keep_going, edge_index))
        keep_going = torch.tanh(self.g3(keep_going, edge_index))
        keep_going = torch.tanh(self.g4(keep_going, edge_index))
        # polling
        keep_going = torch.cat(
            [
                global_max_pool(keep_going, batch_index),
                global_mean_pool(keep_going, batch_index),
            ],
            dim=1,
        )
        keep_going = self.l1(keep_going)
        return keep_going


# print(model)
# print(sum(p.numel() for p in model.parameters()))

model = GraphNeuralNetwork(input_size=data.num_features, hidden_size=64)
loss_fun = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []

for epoch in range(Epochs):
    for i, data in enumerate(train_loader):

        output = model(data.x.float(), data.edge_index, data.batch)
        loss = loss_fun(output, data.y)

        losses.append(loss)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch % 100 == 0:
            print(f"epoch:{epoch} || loss:{loss}")


# visual


losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
loss_indices = [i for i, l in enumerate(losses_float)]
plt.plot(loss_indices, losses_float)
plt.show()


# test
with torch.no_grad():

    for i, data in enumerate(test_loader):
        output = model(data.x.float(), data.edge_index, data.batch)
        df = pandas.DataFrame()
        df["original"] = data.y.tolist()
        df["predicted"] = output.tolist()

df["original"] = df["original"].apply(lambda item: item[0])
df["predicted"] = df["predicted"].apply(lambda item: item[0])


print(df.head())
