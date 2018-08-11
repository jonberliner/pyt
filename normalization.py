from torch import nn


class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups):
        assert num_channels % num_groups == 0
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.num_channels_per_group = self.num_channels // self.num_channels

        self.group_bn = nn.BatchNorm1d(self.num_channels_per_group)

    def forward(self, xs):
        batch_size = xs.shape[0]
        grouped_x = xs.view(batch_size,
                            self.num_groups,
                            self.num_channels_per_group)
        output = self.group_bn(grouped_x)
        return output.view(batch_size, -1)

        return self.group_bn(xs.view(-1,
                                     self.num_groups,
                                     self.num_channels_per_group))
