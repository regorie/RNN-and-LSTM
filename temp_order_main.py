# coding: utf-8
import sys
sys.path.append('..')
from optimizers import SGD
from dataset.random_permut import generate_batch
from models import *
import csv
import argparse
import pickle
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--attempt", "-a", type=int, required=False, default=100)
parser.add_argument("--evalinterval", '-ei', type=int, default=20)
parser.add_argument("--saveparams", '-sp', type=int, default=1)
parser.add_argument("--sequence_length", '-sq', type=int, default=25)
parser.add_argument("--iteration", '-iter', type=int, default=10)
parser.add_argument("--batch_size", '-bs', type=int, default=3)
parser.add_argument("--wordvec_size", '-wvs', type=int, default=100)
parser.add_argument("--hidden_size", '-hs', type=int, default=50)
parser.add_argument("--model", '-m', type=str, default='RNN')
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.01)
args = parser.parse_args()


def grad_norm(grads):
    norm = 0.0
    for key in grads.keys():
        norm += np.sum(grads[key]**2)
    return np.sqrt(norm)

# 하이퍼파라미터 설정
batch_size = args.batch_size
wordvec_size = args.wordvec_size
hidden_size = args.hidden_size  # RNN의 은닉 상태 벡터의 원소 수
time_size = args.sequence_length  # RNN을 펼치는 크기
sequence_length = args.sequence_length
max_iteration = args.iteration

optimizer_params = {'lr':args.learning_rate}


# 모델 생성
if args.model == 'RNN':
    model = RNNLM(vocab_size=100, hidden_size=50, seed=args.seed)
else:
    model = LSTMLM(vocab_size=100, hidden_size=50, seed=args.seed)
optimizer = SGD(optimizer_params)

acc_list = []
loss_list = []
norm_list = []
start = time.time()
for iter in range(max_iteration):
    # train
    train_xs, train_ts = generate_batch(sequence_length=sequence_length, batch_size=batch_size)
    model.reset_state()

    loss = model.forward(train_xs, train_ts)
    grads = model.backward()

    optimizer.update(grads, model.params)
    
    loss_list.append(loss)
    norm_list.append(grad_norm(grads))

    # evaluate
    test_xs, test_ts = generate_batch(sequence_length=sequence_length, batch_size=100)

    correct = 0
    model.reset_state()
    answers = model.generate(test_xs)
    answers = answers[:, -1, :]
    answers = np.argmax(answers,axis=-1)
    pass
    for i, seq in enumerate(test_ts):
        if seq == answers[i]: correct += 1
    acc = correct/test_ts.shape[0]
    print("Iteration {} Accuracy {}".format(iter, acc))
    acc_list.append(acc)

end = time.time()
print("total time taken: ", end-start)

#plt.plot(np.arange(len(acc_list)), acc_list)
#plt.ylim(-0.2,1.2)
#plt.show()

plt.plot(np.arange(len(norm_list)), norm_list)
#plt.ylim(-0.2,1.2)
plt.show()

# 결과 저장
#trainer.plot(None)

"""
with open('../My_Result/Exp_5_epoch'+str(max_epoch)+'('+str(args.attempt)+').csv', 'w+', newline='') as f:
    write = csv.writer(f)
    write.writerow(trainer.ppl_list)

trainer.plot(file_name='../My_Result/Exp_5_epoch'+str(max_epoch)+'('+str(args.attempt)+').png')

# save model params
model.reset_state()

if bool(args.saveparams):
    params = model.params
    with open('../My_Result/Models/Exp_5_epoch'+str(max_epoch)+'('+str(args.attempt)+').pkl', 'wb+') as f:
        pickle.dump(model, f)
"""