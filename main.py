# coding: utf-8
import sys
sys.path.append('..')
from optimizers import SGD
#from optimizers_check_grad_norm import SGD
from trainers import RnnlmTrainer
from dataset import synthesize_data
from dataset.synthesize_data import load_data, vocab_size
from models import *
import csv
import argparse
import pickle
import time

parser = argparse.ArgumentParser()
parser.add_argument("--attempt", "-a", type=int, required=False, default=100)
parser.add_argument("--evalinterval", '-ei', type=int, default=20)
parser.add_argument("--saveparams", '-sp', type=int, default=1)
parser.add_argument("--sequence_length", '-sq', type=int, default=26)
#parser.add_argument("--timestep", '-ts', type=int, default=6)
parser.add_argument("--epoch", '-ep', type=int, default=100)
parser.add_argument("--batch_size", '-bs', type=int, default=3)
parser.add_argument("--wordvec_size", '-wvs', type=int, default=100)
parser.add_argument("--hidden_size", '-hs', type=int, default=50)
parser.add_argument("--model", '-m', type=str, default='RNN')
args = parser.parse_args()



# 하이퍼파라미터 설정
batch_size = args.batch_size
wordvec_size = args.wordvec_size
hidden_size = args.hidden_size  # RNN의 은닉 상태 벡터의 원소 수
time_size = args.sequence_length  # RNN을 펼치는 크기
max_epoch = args.epoch

optimizer_params = {'lr':1.0}

# 학습 데이터 읽기``
train_xs, train_ts = load_data(len=args.sequence_length, is_random=True)
test_xs, test_ts = load_data(len=args.sequence_length, train=False, is_random=True)
#train_ts[:,:-1] = -1
#test_ts[:,:-1] = -1

#train_xs = train_xs[:1000]
#train_ts = train_ts[:1000]

# 모델 생성
if args.model == 'RNN':
    model = RNNLM(vocab_size, wordvec_size, hidden_size)
else:
    model = LSTMLM(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(optimizer_params)
trainer = RnnlmTrainer(model, optimizer)

acc_list = []
start = time.time()
for ep in range(max_epoch):
    epoch=1
    model.reset_state()
    trainer.fit(train_xs, train_ts, epoch, batch_size, time_size, eval_interval=args.evalinterval)

    correct = 0
    model.reset_state()
    answers = model.generate(test_xs)
    answers = answers[:, -1, :]
    answers = np.argmax(answers,axis=-1)
    for i, seq in enumerate(test_ts):
        if seq[-1] == answers[i]: correct += 1
    acc = correct/test_ts.shape[0]
    print("Epoch {} Accuracy {}".format(ep, acc))
    acc_list.append(acc)

end = time.time()
print(end-start)


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