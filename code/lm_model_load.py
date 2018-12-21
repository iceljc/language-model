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

criterion = nn.CrossEntropyLoss()

def run_epoch(model, data, id_2_word, is_train=False, is_test=False, lr=1.0):
  """Runs the model on the given data."""
  if is_train:
    model.train() # train the model
  else:
    model.eval() # test or validate the model

  future_word_num = 1
  epoch_size = ((len(data) // model.batch_size) - future_word_num) // model.num_steps
  start_time = time.time()
  hidden = model.init_hidden()

  costs = 0.0
  iters = 0
  total = 0
  correct = 0
  total_train = 0
  correct_train = 0

  for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps, future_word_num)):

    inputs = Variable(torch.from_numpy(x.astype(np.int64)).transpose(0,1).contiguous()).cuda()

    model.zero_grad() # clear the gradient in previous step

    hidden = repackage_hidden(hidden) # type(hidden) is 'tuple'
    outputs, hidden = model(inputs, hidden)


    targets = Variable(torch.from_numpy(y.astype(np.int64)).transpose(0,1).contiguous()).cuda()
    
    tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))
    # reshape y into a 1-d tensor

    # outputs.view(-1, model.vocab_size).size() = 700 x 10000
    # tt.size() = 700
    inp = torch.squeeze(inputs.view(-1, model.batch_size * model.num_steps))
    out_loss = outputs.view(-1, model.vocab_size)
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
    
    word_inp_print = np.reshape(word_inp, (model.num_steps, model.batch_size)).T
    word_pred_print = np.reshape(word_pred, (model.num_steps, model.batch_size)).T
    word_tt_print = np.reshape(word_tt, (model.num_steps, model.batch_size)).T
    word_id_pred_ = np.reshape(word_id_pred, (model.num_steps, model.batch_size)).T
    word_id_tt_ = np.reshape(word_id_tt, (model.num_steps, model.batch_size)).T

    # print(word_inp_print)
    # print(word_pred_print)
    # print(word_tt_print)
    
    pred_word_id = np.asarray(word_id_pred_)
    target_word_id = np.asarray(word_id_tt_)
    ######

    loss = criterion(out_loss, tt)
    # loss.data[0] -> get the loss value

    costs += loss.data[0] * model.num_steps
    iters += model.num_steps

    # if is_train:

    #   loss.backward()  # backward propagation
    #   torch.nn.utils.clip_grad_norm(model.parameters(), 0.25) # prevent gradient exploding

    #   for name, p in model.named_parameters():
    #     # """if p.requires_grad:
    #     #   print(name, p.data.size()) """
    #     p.data.add_(-lr, p.grad.data) # update the weight and bias
    #   if step % (epoch_size // 10) == 10:
    #     print("{} loss: {:8.5f}".format(step * 1.0 / epoch_size, (costs/iters)))


    if is_test:
      # if (word_pred_print[0][model.num_steps-1] == word_tt_print[0][model.num_steps-1]):
      # print("----------------------")
      # print("input:")
      # print(' '.join(word_inp_print[0]))
      # print("----------------------")
      # print("target:")
      # print(word_tt_print[0][model.num_steps-1])
      # print("----------------------")
      # print("predict:")
      print(word_pred_print[0][model.num_steps-1])
      # print("----------------------")

    # if is_train:
    #   total_train += model.batch_size      
    #   last = pred_word_id.shape[1]-1

    #   for i in range(pred_word_id.shape[0]):
    #     if (pred_word_id[i][last]==target_word_id[i][last]):
    #       correct_train += 1

    if (is_train == False):
      total += model.batch_size
      last = pred_word_id.shape[1]-1

      for i in range(pred_word_id.shape[0]):
        if (pred_word_id[i][last]==target_word_id[i][last]):
          correct += 1

  # if is_train:
  #   train_accuracy = correct_train / total_train * 100
  #   print("accuracy: {:8.2f}".format(train_accuracy))

  if is_test:
    accuracy = correct / total * 100
    # print("Test accuracy: {:8.2f} %".format(accuracy))

  return (costs / iters)



if __name__ == "__main__":
	model = torch.load('lm_model.pt')

	raw_data = reader.ptb_raw_data(data_path='data')
	test_data, word_to_id, id_2_word = raw_data

	vocab_size = len(word_to_id)
	# print('Vocabluary size: {}'.format(vocab_size))


	# model.batch_size = 1 # to make sure we process all the data
	# print('Test Loss: {:8.5f}'.format(run_epoch(model, test_data, id_2_word, False, True)))
  test_loss = run_epoch(model, test_data, id_2_word, False, True)







