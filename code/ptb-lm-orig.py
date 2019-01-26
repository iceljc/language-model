import argparse
import time
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from lm import repackage_hidden, LM_LSTM
import reader
import numpy as np
import os
from torch import optim

parser = argparse.ArgumentParser(description='Simplest LSTM-based language model in PyTorch')
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus')
parser.add_argument('--hidden_size', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--num_steps', type=int, default=35,
                    help='number of LSTM steps')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of LSTM layers')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability')
parser.add_argument('--inital_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--save', type=str,  default='lm_model.pt',
                    help='path to save the final model')
parser.add_argument('--future_word_num', type=int,  default=1,
                    help='predict word number')
args = parser.parse_args()

# num_steps: the number of unrolls.


def savewords(list_data, outfile_name):
  words = np.asarray(list_data)
  mat = np.matrix(words)

  dirName = 'result'
  check_dir = os.path.isdir(dirName)

  if (check_dir==False):
    os.mkdir(dirName)

  outfile = dirName + '/' + outfile_name + '.txt'
  with open(outfile, 'w') as outfile:
    for line in mat:
      np.savetxt(outfile, line, fmt='%s')

""" define the loss function """
criterion = nn.CrossEntropyLoss()

def run_epoch(model, data, id_2_word, is_train=False, is_test=False, lr=1.0):
  """Runs the model on the given data."""
  if is_train:
    model.train() # train the model
  else:
    model.eval() # test or validate the model

  future_word_num = args.future_word_num
  epoch_size = ((len(data) // model.module.batch_size) - future_word_num) // model.module.num_steps
  start_time = time.time()
  hidden = model.module.init_hidden()

  costs = 0.0
  iters = 0
  total = 0
  correct = 0
  total_train = 0
  correct_train = 0

  for step, (x, y) in enumerate(reader.ptb_iterator(data, model.module.batch_size, model.module.num_steps, future_word_num)):

    inputs = Variable(torch.from_numpy(x.astype(np.int64)).transpose(0,1).contiguous()).cuda()
    # print(inputs.size())
    # print(inputs)
    model.zero_grad() # clear the gradient in previous step

    hidden = repackage_hidden(hidden) # type(hidden) is 'tuple'
    outputs, hidden = model(inputs, hidden)

    # outputs = F.sigmoid(outputs);

    targets = Variable(torch.from_numpy(y.astype(np.int64)).transpose(0,1).contiguous()).cuda()
    

    tt = torch.squeeze(targets.view(-1, model.module.batch_size * model.module.num_steps))
    # reshape y into a 1-d tensor

    # index = []
    # for j in range(y.shape[1]-future_word_num+1):
    #   pair = y[:, j:j+future_word_num]
    #   index.append(pair)

    # index_ = np.asarray(index)
    # target_loss = []
    # for i in range(model.module.num_steps):
    #   t = index_[i]
    #   for j in range(model.module.batch_size):
    #     t_ = t[j]
    #     tt = np.zeros(vocab_size, dtype=np.int64)
    #     tt[t_] = 1
    #     target_loss.append(tt)

    # targetLoss = np.asarray(target_loss)
    # targetLoss = Variable(torch.from_numpy(targetLoss).contiguous()).float().cuda()

    # outputs.view(-1, model.vocab_size).size() = 700 x 10000
    # tt.size() = 700
    inp = torch.squeeze(inputs.view(-1, model.module.batch_size * model.module.num_steps))
    out_loss = outputs.view(-1, model.module.vocab_size)
    max_val, index = torch.max(out_loss, dim=1)

    # ######
    word_inp = []
    word_pred = []
    word_tt = []
    word_id_pred = []
    word_id_tt = []

    for i in range(list(index.size())[0]):
      ind_inp = inp.data[i]
      w_inp = id_2_word[ind_inp]
      word_inp.append(w_inp)

      ind_pred = list(index.data[i])[0]
      w_pred = id_2_word[ind_pred]
      word_pred.append(w_pred)
      word_id_pred.append(ind_pred)

      ind_tt = tt.data[i]
      w_tt = id_2_word[ind_tt]
      word_tt.append(w_tt)
      word_id_tt.append(ind_tt)
    
    # word_inp_print = np.reshape(word_inp, (model.num_steps, model.batch_size)).T
    # word_pred_print = np.reshape(word_pred, (model.num_steps, model.batch_size)).T
    # word_tt_print = np.reshape(word_tt, (model.num_steps, model.batch_size)).T
    word_id_pred_ = np.reshape(word_id_pred, (model.module.num_steps, model.module.batch_size)).T
    word_id_tt_ = np.reshape(word_id_tt, (model.module.num_steps, model.module.batch_size)).T
    pred_word_id = np.asarray(word_id_pred_)
    target_word_id = np.asarray(word_id_tt_)
    ######

    loss = criterion(out_loss, tt)
    # loss.data[0] -> get the loss value

    costs += loss.data[0] * model.module.num_steps
    iters += model.module.num_steps

    if is_train:
      optimizer.zero_grad()
      loss.backward()  # backward propagation
      torch.nn.utils.clip_grad_norm(model.module.parameters(), 0.25) # prevent gradient exploding
      optimizer.step()
      # for name, p in model.named_parameters():
      #   # """if p.requires_grad:
      #   #   print(name, p.data.size()) """
      #   p.data.add_(-lr, p.grad.data) # update the weight and bias
      if step % (epoch_size // 10) == 10:
        print("{} loss: {:8.5f}".format(step * 1.0 / epoch_size, (costs/iters)))
        # print("{} perplexity: {:8.2f} speed: {} wps".format(step * 1.0 / epoch_size, np.exp(costs / iters),
        #                                                iters * model.batch_size / (time.time() - start_time)))
        
        # print("input:")
        # print(word_inp_print)
        # print("----------------------")
        # print("predict:")
        # print(word_pred_print)
        # print("----------------------")
        # print("target:")
        # print(word_tt_print)

      # savewords(word_inp_print, 'input_train')
      # savewords(word_pred_print, 'predict_train')
      # savewords(word_tt_print, 'target_train')
    # elif is_test:
    #   savewords(word_inp_print, 'input_test')
    #   savewords(word_pred_print, 'predict_test')
    #   savewords(word_tt_print, 'target_test')

    
    if is_train:
      total_train += model.module.batch_size      
      last = pred_word_id.shape[1]-1

      for i in range(pred_word_id.shape[0]):
        if (pred_word_id[i][last]==target_word_id[i][last]):
          correct_train += 1

    if (is_train == False):
      total += model.module.batch_size
      last = pred_word_id.shape[1]-1

      for i in range(pred_word_id.shape[0]):
        if (pred_word_id[i][last]==target_word_id[i][last]):
          correct += 1

  if is_train:
    accuracy = correct_train / total_train * 100
    print("accuracy: {:8.2f}".format(accuracy))

  if (is_train == False):
    accuracy = correct / total * 100
    print("accuracy: {:8.2f}".format(accuracy))

  return accuracy, (costs / iters)
  # return np.exp(costs / iters) 


if __name__ == "__main__":
  
  raw_data = reader.ptb_raw_data(data_path=args.data)
  train_data, valid_data, test_data, word_to_id, id_2_word = raw_data


  vocab_size = len(word_to_id)
  print('Vocabluary size: {}'.format(vocab_size))
  
  # define the model
  model = LM_LSTM(embedding_dim=args.hidden_size, num_steps=args.num_steps, batch_size=args.batch_size,
                  vocab_size=vocab_size, num_layers=args.num_layers, dp_keep_prob=args.dp_keep_prob)

  os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"
  model = nn.DataParallel(model, dim=1)
  model.cuda() # move the model to gpu

  
  lr = args.inital_lr
  # decay factor for learning rate
  lr_decay_base = 1 / 1.15
  # we will not touch lr for the first m_flat_lr epochs
  m_flat_lr = 14.0
  params = filter(lambda p:p.requires_grad, model.module.parameters())
  optimizer = optim.Adam(params, lr=1e-4, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)

  print("########## Training ##########################")
  train_loss = []
  valid_loss = []
  train_accuracy = []
  valid_accuracy = []
  for epoch in range(args.num_epochs):
    lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
    lr = lr * lr_decay # decay lr if it is time
    train_acc, train_p = run_epoch(model, train_data, id_2_word, True, False, lr)
    print('Train perplexity at epoch {}: {:8.5f}'.format(epoch, train_p))
    train_loss.append(train_p)
    train_accuracy.append(train_acc)

    valid_acc, valid_p = run_epoch(model, valid_data, id_2_word)
    print('Validation perplexity at epoch {}: {:8.5f}'.format(epoch, valid_p))
    valid_loss.append(valid_p)
    valid_accuracy.append(valid_acc)

  with open("result/train_loss.txt", "w") as f:
    for item in train_loss:
      f.write("%s\n" % item)

  with open("result/valid_loss.txt", "w") as f:
    for item in valid_loss:
      f.write("%s\n" % item)

  with open("result/train_acc.txt", "w") as f:
    for item in train_accuracy:
      f.write("%s\n" % item)

  with open("result/valid_acc.txt", "w") as f:
    for item in valid_accuracy:
      f.write("%s\n" % item)
  
  print("########## Testing ##########################")
  model.batch_size = 1 # to make sure we process all the data
  test_acc, test_loss = run_epoch(model, test_data, id_2_word, False, True)
  print('Test Perplexity: {:8.5f}'.format(test_loss))
  with open(args.save, 'wb') as f:
    torch.save(model, f)
  print("########## Done! ##########################")



