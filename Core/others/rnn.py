import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchtext.data import Field
from torch.utils.tensorboard import SummaryWriter

import random
from typing import Tuple



class Encoder(nn.Module):
    """
        input_dim:  the source vocabulary size
        emd_dim:   the source embedding feature dimension
        enc_hid_dim:  the encoder hidden feature dimension
        dec_hid_dim:  the decoder hidden feature dimension
        dropout:  dropout ratio
    """

    def __init__(self, input_dim, emd_dim, enc_hid_dim,
                 dec_hid_dim, dropout: float = 0.5):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emd_dim = emd_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=self.input_dim, embedding_dim=self.emd_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.rnn = nn.GRU(input_size=self.emd_dim,
                          hidden_size=self.enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(self.enc_hid_dim*2, self.dec_hid_dim)

    def forward(self, src: Tensor) -> Tuple[Tensor]:
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden


class Attention(nn.Module):
    """
        atten_dim: the energy vector dimension
    """

    def __init__(self, enc_hid_dim, dec_hid_dim, atten_dim):
        super(Attention, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.atten_dim = atten_dim

        self.atten = nn.Linear(
            self.enc_hid_dim*2+self.dec_hid_dim, self.atten_dim)

    def forward(self, encoder_outputs, decoder_hidden):
        src_len = encoder_outputs.shape[0]
        repeated_dec_hid = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.atten(
            torch.cat((encoder_outputs, repeated_dec_hid), dim=2)))
        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
        output_dim:  the target vocabulary size
        emd_dim:  the target embedding feature dimension
        enc_hid_dim:  the encoder hidden feature dimension
        dec_hid_dim:  the decoder hidden feature dimension
        dropout:  dropout ratio
    """

    def __init__(self, output_dim, emd_dim, enc_hid_dim, dec_hid_dim,
                 dropout: float,
                 attention: nn.Module):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.emd_dim = emd_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emd_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emd_dim + enc_hid_dim*2, dec_hid_dim)
        self.out = nn.Linear(dec_hid_dim + emd_dim + enc_hid_dim*2, output_dim)

    def _weighted_encoder_rep(self, encoder_outputs, decoder_hidden):
        a = self.attention(encoder_outputs, decoder_hidden)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # matrix multiply 代替 element-wise multiply
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(self, decoder_input, encoder_outputs, decoder_hidden):
        decoder_input = decoder_input.unsqueeze(0)
        embedded = self.dropout(self.embedding(decoder_input))
        weighted_encoder_rep = self._weighted_encoder_rep(
            encoder_outputs, decoder_hidden)
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        output, decoder_hidden = self.rnn(
            rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        output = self.out(
            torch.cat((output, embedded, weighted_encoder_rep), dim=1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    """
        sequence to sequence forward neuron network
    """

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, device, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        max_seq_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_seq_len, batch_size,
                              trg_vocab_size).to(device)
        encoder_outputs, hidden = self.encoder(src)
        # first input to the decoder is the '<sos>'
        decoder_input = trg[0, :]

        for t in range(1, max_seq_len):
            output, hidden = self.decoder(
                decoder_input, encoder_outputs, hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = trg[t] if teacher_force else top1

        return outputs