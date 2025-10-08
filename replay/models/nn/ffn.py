import contextlib

import torch


class PointWiseFeedForward(torch.nn.Module):
    """
    Point wise feed forward network layer

    Link: https://arxiv.org/pdf/1808.09781.pdf
    """

    def __init__(self, hidden_size: int, dropout: float) -> None:
        """
        :param hidden_size: Hidden size.
        :param dropout: Dropout rate.
        """
        super().__init__()

        self.conv1 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout)

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(self, inputs: torch.LongTensor) -> torch.LongTensor:
        """
        :param inputs: Query feature vector.

        :returns: Output tensors.
        """
        outputs = self.dropout2(self.conv2(self.dropout1(self.relu(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs

        return outputs
