import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LM_LSTM(nn.Module):
  """Simple LSTM-based language model"""
  def __init__(self, embedding_dim, num_steps, batch_size, vocab_size, num_layers, dp_keep_prob):
    super(LM_LSTM, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_steps = num_steps
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.dp_keep_prob = dp_keep_prob
    self.num_layers = num_layers
    self.dropout = nn.Dropout(1 - dp_keep_prob)
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=embedding_dim,
                            num_layers=num_layers,
                            batch_first = False,
                            dropout=1 - dp_keep_prob)
    # self.pre_sm_fc = nn.Linear(in_features=embedding_dim*2,
    #                        out_features=embedding_dim)
    self.sm_fc = nn.Linear(in_features=embedding_dim,
                           out_features=vocab_size)
    self.init_weights()

  def init_weights(self):
    init_range = 0.1
    self.word_embeddings.weight.data.uniform_(-init_range, init_range)
    self.sm_fc.bias.data.fill_(0.0)
    self.sm_fc.weight.data.uniform_(-init_range, init_range)

  def init_hidden(self):
    weight = next(self.parameters()).data # retrieve the first parameter from the class
    return (Variable(weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_()),
            Variable(weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_()))

  def forward(self, inputs, hidden):
    embeds = self.word_embeddings(inputs)

    embeds = self.dropout(embeds) 
    # embedding layer weight size: (vocab_size x embedding_dim) 10000 x 1500
    # dim of embeds: (num_steps x batch_size x embedding_dim) 35 x 20 x 1500

    lstm_out, hidden = self.lstm(embeds)
    # lstm_out = torch.transpose(lstm_out, 0,1)
    # hidden: (h, c)
    # dim of hidden: (num_layers x batch_size x embedding_dim) 2 x 20 x 1500
    # dim of lstm_out: (num_steps x batch_size x embedding_dim) 35 x 20 x 1500

    lstm_out = self.dropout(lstm_out)

    logits = self.sm_fc(lstm_out.view(-1, self.embedding_dim))

    # logits = self.pre_sm_fc(lstm_out.view(-1, self.embedding_dim*2))
    # logits = self.sm_fc(self.dropout(F.tanh(logits)))
    # dim of logits: (num_steps x batch_size x vocab_size) 700 x 10000

    return logits.view(self.num_steps, -1, self.vocab_size), hidden
    # embeds = self.word_embeddings(inputs.view(inputs.size(1), 1))
    # embeds = self.dropout(embeds)
    # embeds = embeds.view(1, embeds.size(0), embeds.size(2))
    # lstm_out, hidden = self.lstm(embeds, hidden)
    # lstm_out = self.dropout(lstm_out)
    # logits = self.sm_fc(lstm_out.view(inputs.size(1), self.embedding_dim))
    # return logits.view(1, inputs.size(1), -1), hidden


def repackage_hidden(h):
  """Wraps hidden states in new Variables, to detach them from their history."""
  if type(h) == Variable:
    return Variable(h.data)
  else:
    return tuple(repackage_hidden(v) for v in h)

