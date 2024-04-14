import torch.nn as nn
import torch


class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes.
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]  # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input, vocab):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input, None

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], vocab, total_width)

            return input

    def transform_slice(self, e, vocab, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0
                vocab[bgn : bgn + distance, :, :] = 0


class SpecAugmentation(nn.Module):
    def __init__(
        self, time_drop_width, time_stripes_num, freq_drop_width, freq_stripes_num
    ):
        """Spec augmetation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(
            dim=2, drop_width=time_drop_width, stripes_num=time_stripes_num
        )

        self.freq_dropper = DropStripes(
            dim=3, drop_width=freq_drop_width, stripes_num=freq_stripes_num
        )

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x

def dropmask(
    x,
    vocab,
    channel_stripes_num=2,
    channel_drop_width=128,
    temporal_stripes_num=2,
    temporal_drop_width=4,
):
    total_width = vocab.shape[0]
    channel_mask = torch.zeros(vocab.shape[0], device=vocab.device)
    for _ in range(channel_stripes_num):
        distance = torch.randint(low=0, high=channel_drop_width, size=(1,))[0]
        bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]
        channel_mask[bgn : bgn + distance] = 1

    total_width = x.shape[2]
    temporal_mask = torch.zeros(x.shape[2], device=x.device)
    if temporal_drop_width != 0:
        for _ in range(temporal_stripes_num):
            distance = torch.randint(low=0, high=temporal_drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]
            temporal_mask[bgn : bgn + distance] = 1

    return (x.masked_fill(channel_mask == 1, 0)).permute(0, 1, 3, 2).masked_fill(
        temporal_mask == 1, 0
    ).permute(0, 1, 3, 2), vocab.permute(1, 2, 0).masked_fill(
        channel_mask == 1, 0
    ).permute(
        2, 0, 1
    )
