import os
import time
import torch as torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from models import TxtModule, ImgModule
from data.data_loader import *
from utils.evaluate import *
from utils.generatecode import *
from utils.calculate import *
from loss.multisimilarity import *
from loss.quantizationloss import *

from torch.autograd import Variable
from torch.optim import SGD

from tqdm import tqdm
import fire


fire.Fire()
# config:
isvalid = True
num_query = 2000
isample = True
num_samples2 = 2000

root = './data/FLICKR-25K.mat'
pretrain_model_path = './data/imagenet-vgg-f.mat'
# Flickr25K
images, tags, labels = load_data(root)
pretrain_model = load_pretrain_model(pretrain_model_path)
num_seen = 22
code_length = 64
lr1 = 0.001
lr2 = 0.0000001
max_epoch = 200
batch_size = 128
device = True
gamma = 10
eta = 10
mu = 1
alpha = 10
beta = 100
tag_length = tags.shape[1]
label_length = labels.shape[1]

#  加载数据并划分
#  Flickr25K
X, Y, L = split_data(images, tags, labels, num_query, num_seen, seed=None)

seen_L = torch.from_numpy(L['seen'])
seen_x = torch.from_numpy(X['seen'])
seen_y = torch.from_numpy(Y['seen'])

unseen_L = torch.from_numpy(L['unseen'])
unseen_x = torch.from_numpy(X['unseen'])
unseen_y = torch.from_numpy(Y['unseen'])

query_L = torch.from_numpy(L['query'])
query_x = torch.from_numpy(X['query'])
query_y = torch.from_numpy(Y['query'])

retrieval_L = torch.from_numpy(L['retrieval'])
retrieval_x = torch.from_numpy(X['retrieval'])
retrieval_y = torch.from_numpy(Y['retrieval'])
print('...loading and splitting data finish')

#  初始化：定义网络模型、优化器、学习率
img_model = ImgModule(code_length, pretrain_model)
txt_model = TxtModule(tag_length, code_length)
label_model = TxtModule(label_length, code_length)
if device:
    img_model = img_model.cuda()
    txt_model = txt_model.cuda()
    label_model = label_model.cuda()
optimizer_img1 = SGD(img_model.parameters(), lr=lr1)
optimizer_txt1 = SGD(txt_model.parameters(), lr=lr1)
optimizer_label = SGD(label_model.parameters(), lr=lr1)

learning_rate1 = np.linspace(lr1, np.power(10, -10.), max_epoch + 1)
result = {
    'loss': []
}
#  初始化变量
num_seendata = len(seen_L)
#  采样训练数据
num_train = seen_L.shape[0]

F_buffer = torch.randn(num_train, code_length)
G_buffer = torch.randn(num_train, code_length)
L_buffer = torch.randn(num_train, code_length)

if device:
    seen_L = seen_L.cuda()
    F_buffer = F_buffer.cuda()
    G_buffer = G_buffer.cuda()
    L_buffer = L_buffer.cuda()

old_F = torch.randn(num_seendata, code_length).sign()
old_G = torch.randn(num_seendata, code_length).sign()
Bx = torch.sign(F_buffer + L_buffer)
By = torch.sign(G_buffer + L_buffer)
ones = torch.ones(batch_size, 1)
ones_ = torch.ones(num_train - batch_size, 1)
unupdated_size = num_train - batch_size
if device:
    old_F = old_F.cuda()
    old_G = old_G.cuda()

max_mapi2t = max_mapt2i = 0.
total_time_o = time.time()
for epoch in range(max_epoch):
    # train label net
    print("\n...start to train label net")
    for i in tqdm(range(num_train // batch_size)):
        index = np.random.permutation(num_train)
        ind = index[0: batch_size]
        unupdated_ind = np.setdiff1d(range(num_train), ind)

        sample_L = Variable(seen_L[ind, :])
        label = seen_L[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
        label = Variable(label)
        if device:
            sample_L = sample_L.cuda()
            ones = ones.cuda()
            ones_ = ones_.cuda()
            label = label.cuda()

        cur_l = label_model(label)
        L_buffer[ind, :] = cur_l.data
        F = Variable(F_buffer)
        G = Variable(G_buffer)
        H = Variable(L_buffer)

        KLloss_ll = eta * multilabelsimilarityloss(sample_L, seen_L, cur_l, H)
        KLloss_lx = multilabelsimilarityloss(sample_L, seen_L, cur_l, F)
        KLloss_ly = gamma * multilabelsimilarityloss(sample_L, seen_L, cur_l, G)
        quantization_l = (mu * quantizationLoss(cur_l, Bx[ind, :]) + mu * quantizationLoss(cur_l, By[ind, :])) / 2

        loss_l = KLloss_ll + KLloss_lx + KLloss_ly + quantization_l

        optimizer_label.zero_grad()
        loss_l.backward()
        optimizer_label.step()

    # train image net
    print("\n...start to train image net")
    for i in tqdm(range(num_train // batch_size)):
        index = np.random.permutation(num_train)
        ind = index[0: batch_size]
        unupdated_ind = np.setdiff1d(range(num_train), ind)

        sample_L = Variable(seen_L[ind, :])
        image = Variable(seen_x[ind].type(torch.float))
        image = Variable(image)
        if device:
            image = image.cuda()
            sample_L = sample_L.cuda()
            ones = ones.cuda()
            ones_ = ones_.cuda()

        cur_f = img_model(image)
        F_buffer[ind, :] = cur_f.data
        F = Variable(F_buffer)
        H = Variable(L_buffer)

        KLloss_xx = eta * multilabelsimilarityloss(sample_L, seen_L, cur_f, F)
        KLloss_xl = multilabelsimilarityloss(sample_L, seen_L, cur_f, H)
        quantization_x1 = mu * quantizationLoss(cur_f, Bx[ind, :])

        loss_x1 = KLloss_xx + KLloss_xl + quantization_x1

        optimizer_img1.zero_grad()
        loss_x1.backward()
        optimizer_img1.step()

    # train txt net
    print("\n...start to train txt net")
    for i in tqdm(range(num_train // batch_size)):
        index = np.random.permutation(num_train)
        ind = index[0: batch_size]
        unupdated_ind = np.setdiff1d(range(num_train), ind)

        sample_L = Variable(seen_L[ind, :])
        text = seen_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
        text = Variable(text)
        if device:
            text = text.cuda()
            sample_L = sample_L.cuda()

        cur_g = txt_model(text)
        G_buffer[ind, :] = cur_g.data
        H = Variable(L_buffer)
        G = Variable(G_buffer)

        KLloss_yy = eta * multilabelsimilarityloss(sample_L, seen_L, cur_g, G)
        KLloss_yl = multilabelsimilarityloss(sample_L, seen_L, cur_g, H)
        quantization_y1 = mu * quantizationLoss(cur_g, By[ind, :])

        loss_y1 = KLloss_yy + KLloss_yl + quantization_y1

        optimizer_txt1.zero_grad()
        loss_y1.backward()
        optimizer_txt1.step()

    # update B
    Bx = torch.sign(2 * F_buffer + L_buffer)
    By = torch.sign(2 * G_buffer + L_buffer)

    print('...epoch: %3d, LabelLoss:%3.3f, ImgLoss:%3.3f, TxtLoss:%3.3f,lr: %.10f' % (
        epoch + 1, loss_l, loss_x1, loss_y1, lr1))

    if isvalid:
        if device:
            seen_L = seen_L.cuda()
            query_L = query_L.cuda()
            retrieval_L = retrieval_L.cuda()
        qBX = generate_image_code(img_model, query_x, code_length, batch_size, device)
        qBY = generate_text_code(txt_model, query_y, code_length, batch_size, device)
        rBX = generate_image_code(img_model, retrieval_x, code_length, batch_size, device)
        rBY = generate_text_code(txt_model, retrieval_y, code_length, batch_size, device)
        mapi2t = mean_average_precision(qBX, rBY, query_L, retrieval_L, device)
        mapt2i = mean_average_precision(qBY, rBX, query_L, retrieval_L, device)
        print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
        if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
            max_mapi2t = mapi2t
            max_mapt2i = mapt2i
            img_model.save(img_model.module_name + '.pth')
            txt_model.save(txt_model.module_name + '.pth')

    lr1 = learning_rate1[epoch + 1]

    # set learning rate
    for param in optimizer_label.param_groups:
        param['lr'] = lr1
    for param in optimizer_img1.param_groups:
        param['lr'] = lr1
    for param in optimizer_txt1.param_groups:
        param['lr'] = lr1
o_time = time.time() - total_time_o
print('...training original phase finish time: %3.2f' % o_time)
# Save checkpoints 保存原始哈希码
torch.save(Bx.cpu(), os.path.join('checkpoints', 'old_F.t'))
torch.save(By.cpu(), os.path.join('checkpoints', 'old_G.t'))

if isvalid:
    print('   original-max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    result['mapi2t'] = max_mapi2t
    result['mapt2i'] = max_mapt2i
else:
    mapi2t, mapt2i = valid(img_model, txt_model, query_x, F, query_y, G,
                           query_L, retrieval_L, code_length, batch_size, device)
    print('   original-max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
    result['mapi2t'] = mapi2t
    result['mapt2i'] = mapt2i

# 增量部分
# 加载原始哈希码
old_F = torch.load(os.path.join('checkpoints', 'old_F.t'))
old_G = torch.load(os.path.join('checkpoints', 'old_G.t'))

# 重新定义优化器和学习率衰减策略
optimizer_img2 = torch.optim.Adam(img_model.parameters(), lr=lr2)
optimizer_txt2 = torch.optim.Adam(txt_model.parameters(), lr=lr2)

lr_scheduler_img2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_img2, 0.95)
lr_scheduler_txt2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_txt2, 0.95)

# 初始化变量
num_retrieval = len(retrieval_L)
num_unseendata = len(unseen_x)

F_out = torch.randn(num_samples2, code_length)
G_out = torch.randn(num_samples2, code_length)

Ux_new = torch.zeros(num_samples2, code_length)
Uy_new = torch.zeros(num_samples2, code_length)

new_F_buffer = torch.randn(num_samples2, code_length)
new_G_buffer = torch.randn(num_samples2, code_length)

new_F = torch.randn(num_unseendata, code_length).sign()
new_G = torch.randn(num_unseendata, code_length).sign()
F2 = torch.cat((old_F, new_F), dim=0)
G2 = torch.cat((old_G, new_G), dim=0)
if device:
    Ux_new = Ux_new.cuda()
    Uy_new = Uy_new.cuda()
    old_F = old_F.cuda()
    new_F = new_F.cuda()
    old_G = old_G.cuda()
    new_G = new_G.cuda()
    F2 = F2.cuda()
    G2 = G2.cuda()
    new_F_buffer = new_F_buffer.cuda()
    new_G_buffer = new_G_buffer.cuda()

max_mapi2t = max_mapt2i = 0.

train_L, train_x, train_y, train_index, unseen_in_unseen_index, unseen_in_sample_index = sample_data(retrieval_x, retrieval_y, retrieval_L, num_seendata, num_samples2)
if device:
    train_L = train_L.cuda()
    retrieval_L = retrieval_L.cuda()
    seen_L = seen_L.cuda()
    unseen_L = unseen_L.cuda()

total_time_i = time.time()
for epoch in range(max_epoch):
    old_S = multi_similar(train_L, seen_L, device)
    new_S = multi_similar(train_L, unseen_L, device)
    S = torch.cat((old_S, new_S), dim=1)
    # train image net
    print("...start to train image net")
    for i in tqdm(range(num_samples2 // batch_size)):
        index = np.random.permutation(num_samples2)
        ind = index[0: batch_size]
        unupdated_ind = np.setdiff1d(range(num_samples2), ind)

        sample_L = Variable(train_L[ind, :])
        image = train_x[ind].type(torch.float)
        image = Variable(image)
        if device:
            image = image.cuda()
            sample_L = sample_L.cuda()

        cur_f = img_model(image)
        new_F_buffer[ind, :] = cur_f.data
        F_out = Variable(new_F_buffer)
        Ux_new = Variable(torch.tanh(new_F_buffer))

        hashloss_x1 = ((code_length * old_S[ind, :] - cur_f @ old_F.t()) ** 2).sum()
        hashloss_x2 = ((code_length * new_S[ind, :] - cur_f @ new_F.t()) ** 2).sum()
        quantization_x2 = torch.sum(torch.pow(F2[ind, :] - cur_f, 2))
        balance_x2 = (cur_f @ torch.ones(cur_f.shape[1], 1, device=cur_f.device)).sum()
        loss_x2 = hashloss_x1 + alpha * quantization_x2 + beta * balance_x2
        loss_x2 /= (batch_size * F2.shape[0])

        optimizer_img2.zero_grad()
        loss_x2.backward()
        optimizer_img2.step()
    # update F
    expand_Ux_new = torch.zeros(num_unseendata, code_length).cuda()
    expand_Ux_new[unseen_in_unseen_index, :] = Ux_new[unseen_in_sample_index, :]
    new_F = solve_dcc(new_F, Ux_new, expand_Ux_new, S[:, num_seendata:], code_length, alpha)
    F2 = torch.cat((old_F, new_F), dim=0).cuda()
    # train txt net
    print("...start to train txt net")
    for i in tqdm(range(num_samples2 // batch_size)):
        index = np.random.permutation(num_samples2)
        ind = index[0: batch_size]
        unupdated_ind = np.setdiff1d(range(num_samples2), ind)

        sample_L = Variable(train_L[ind, :])
        text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
        text = Variable(text)
        if device:
            text = text.cuda()
            sample_L = sample_L.cuda()

        cur_g = txt_model(text)
        new_G_buffer[ind, :] = cur_g.data
        G_out = Variable(new_G_buffer)
        Uy_new = Variable(torch.tanh(new_G_buffer))

        hashloss_y1 = ((code_length * old_S[ind, :] - cur_g @ old_G.t()) ** 2).sum()
        hashloss_y2 = ((code_length * new_S[ind, :] - cur_g @ new_G.t()) ** 2).sum()
        quantization_y2 = torch.sum(torch.pow(G2[ind, :] - cur_g, 2))
        balance_y2 = (cur_g @ torch.ones(cur_g.shape[1], 1, device=cur_g.device)).sum()
        loss_y2 = hashloss_y1 + alpha * quantization_y2 + beta * balance_y2
        loss_y2 = loss_y2 / (batch_size * G2.shape[0])

        optimizer_txt2.zero_grad()
        loss_y2.backward()
        optimizer_txt2.step()

    # update G
    expand_Uy_new = torch.zeros(num_unseendata, code_length).cuda()
    expand_Uy_new[unseen_in_unseen_index, :] = Uy_new[unseen_in_sample_index, :]
    new_G = solve_dcc(new_G, Uy_new, expand_Uy_new, S[:, num_seendata:], code_length, alpha)
    G2 = torch.cat((old_G, new_G), dim=0).cuda()

    update_B_new = torch.sign(F2 + G2)
    # calculate total loss
    loss2 = calc_increment_loss(Ux_new, Uy_new, update_B_new, S, code_length, train_index, alpha, beta)

    print('...epoch: %3d, loss: %3.3f, lr: %.10f' % (epoch + 1, loss2.data, optimizer_img2.param_groups[0]['lr']))

    if isvalid:
        database_L = torch.cat((seen_L, unseen_L), dim=0)
        if device:
            query_L = query_L.cuda()
            database_L = database_L.cuda()
        qBX2 = generate_image_code(img_model, query_x, code_length, batch_size, device)
        qBY2 = generate_text_code(txt_model, query_y, code_length, batch_size, device)

        mapi2t = mean_average_precision(qBX2, update_B_new, query_L, database_L, device)
        mapt2i = mean_average_precision(qBY2, update_B_new, query_L, database_L, device)
        print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))

        if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
            max_mapi2t = mapi2t
            max_mapt2i = mapt2i
            img_model.save(img_model.module_name + '.pth')
            txt_model.save(txt_model.module_name + '.pth')

    lr_scheduler_img2.step()
    lr_scheduler_txt2.step()
i_time = time.time() - total_time_i
print('...training incremental phase finish and time : %3.2f' % i_time)
if isvalid:
    print(' DLCH max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    result['mapi2t'] = max_mapi2t
    result['mapt2i'] = max_mapt2i
