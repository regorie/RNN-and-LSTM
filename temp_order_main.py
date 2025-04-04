# coding: utf-8
import sys
sys.path.append('..')
from optimizers import SGD
from dataset.temp_order import generate_batch
from models import *
import csv
import argparse
import pickle
import time
import matplotlib.pyplot as plt
from clip_grad import clip_grads

parser = argparse.ArgumentParser()
parser.add_argument("--exp_no", "-en", type=int, required=True)
parser.add_argument("--evalinterval", '-ei', type=int, default=20)
parser.add_argument("--saveparams", '-sp', type=int, default=1)
parser.add_argument("--sequence_length", '-sq', type=int, default=25)
parser.add_argument("--state_reset", '-srt', type=int, default=1)
parser.add_argument("--iteration", '-iter', type=int, default=10)
parser.add_argument("--test_iteration", '-titer', type=int, default=2560)
parser.add_argument("--batch_size", '-bs', type=int, default=20)
parser.add_argument("--test_batch_size", '-tbs', type=int, default=10000)
parser.add_argument("--hidden_size", '-hs', type=int, default=50)
parser.add_argument("--model", '-m', type=str, default='RNN')
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.01)
parser.add_argument("--clip_grad", '-clip', type=float, default=1.0)
parser.add_argument("--target_accuracy", '-ta', type=float, default=0.99)
args = parser.parse_args()


# 하이퍼파라미터 설정
batch_size = args.batch_size
hidden_size = args.hidden_size  # RNN의 은닉 상태 벡터의 원소 수
time_size = args.sequence_length  # RNN을 펼치는 크기
sequence_length = args.sequence_length
max_iteration = args.iteration
eval_interval = args.evalinterval

if args.state_reset == 0:
    stateful = False
else:
    stateful = True

optimizer_params = {'lr':args.learning_rate}


# 모델 생성
if args.model == 'RNN':
    model = RNN_manyToOne(nin=8, nout=4, hidden_size=50, scale=0.1, stateful=stateful, seed=args.seed)
elif args.model == 'LSTM':
    model = LSTM_manyToOne(nin=8, nout=4, hidden_size=50, scale=0.1, stateful=stateful, seed=args.seed)
elif args.model == 'LSTM97':
    model = LSTM97_manyToOne(nin=8, nout=4, num_block=2, num_cell_per_block=2, all_label=8, scale=0.1, seed=args.seed)
optimizer = SGD(optimizer_params)

results = []
norm_sum = 0.0
loss_sum = 0.0
start = time.time()
for iter in range(max_iteration//eval_interval):
    for inneriter in range(eval_interval):
        # train
        train_xs, train_ts = generate_batch(seq_length_range=[100, 110], 
                                            pos0_range=[10, 20],
                                            pos1_range=[50, 60],
                                            label_num=8,
                                            batch_size=batch_size)

        loss_sum += model.forward(train_xs, train_ts)
        grads = model.backward()

        if args.clip_grad > 0.0:
            norm_sum += clip_grads(grads, args.clip_grad)

        optimizer.update(grads, model.params)

    ##### evaluate #####
    if args.state_reset == 2:
        hidden_state = model.final_h
        if hasattr(model, "final_c"): cell_state = model.final_c

    for testiter in range(args.test_iteration):
        test_xs, test_ts = generate_batch(seq_length_range=[100, 110], 
                                        pos0_range=[10, 20],
                                        pos1_range=[50, 60],
                                        label_num=8,
                                        batch_size=args.test_batch_size)

        correct = 0
        model.reset_state()
        answers = model.predict(test_xs)
        answers = np.argmax(answers,axis=-1)
        for i, seq in enumerate(test_ts):
            if seq == answers[i]: correct += 1
    acc = correct/test_ts.shape[0]
    print("Iteration {} Accuracy {}".format(iter*eval_interval, acc))

    results.append([acc, loss_sum/eval_interval, norm_sum/eval_interval])
    loss_sum = 0.0
    norm_sum = 0.0

    if acc >= 0.99 : break

    model.reset_state()
    if args.state_reset== 2:
        model.final_h = hidden_state
        if hasattr(model, "final_c"): model.final_c = cell_state

end = time.time()
print("total time taken: ", end-start)


with open('./Result/results({}).csv'.format(args.exp_no), 'w+', newline='') as f:
    write = csv.writer(f)
    write.writerows(results)

if args.saveparams == 1:
    params = model.params
    with open('./Result/Models/model_params({}).pkl'.format(args.exp_no), 'wb+') as f:
        pickle.dump(params, f)