# ======================================
# === Pytorch手写Transformer完整代码
# ======================================
"""
code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612, modify by shwei
Reference: https://github.com/jadore801120/attention-is-all-you-need-pytorch
           https://github.com/JayParks/transformer
"""
# ====================================================================================================
# 数据构建
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from transformer_net import  *

# device = 'cpu'
device = 'cuda'

# transformer epochs
epochs = 100
# epochs = 1000




# ==============================================================================================
# 数据构建


def make_data(sentences):
    """把单词序列转换为数字序列"""
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

        # [[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 8, 4, 9, 6, 7, 0], [1, 2, 3, 4, 10, 6, 7, 0]]
        enc_inputs.extend(enc_input)
        # [[9, 1, 2, 3, 4, 5, 11], [9, 1, 2, 6, 7, 5, 11], [9, 1, 2, 3, 8, 5, 11]]
        dec_inputs.extend(dec_input)
        # [[1, 2, 3, 4, 5, 11, 10], [1, 2, 6, 7, 5, 11, 10], [1, 2, 3, 8, 5, 11, 10]]
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)





class MyDataSet(Data.Dataset):
    """自定义DataLoader"""

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]











def main():
    # 训练集
    sentences = [
        # 中文和英语的单词个数不要求相同
        # enc_input                dec_input           dec_output
        ['我 有 一 个 好 朋 友 P', 'S I have a good friend .', 'I have a good friend . E'],
        ['我 有 零 个 女 朋 友 P', 'S I have zero girl friend .', 'I have zero girl friend . E'],
        ['我 有 一 个 男 朋 友 P', 'S I have a boy friend .', 'I have a boy friend . E']
    ]
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    loader = Data.DataLoader(
        MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)


    model = Transformer().to(device)
    # 这里的损失函数里面设置了一个参数 ignore_index=0，因为 "pad" 这个单词的索引为 0，这样设置以后，就不会计算 "pad" 的损失（因为本来 "pad" 也没有意义，不需要计算）
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-3,
                          momentum=0.99)  # 用adam的话效果不好

    # ====================================================================================================
    for epoch in range(epochs):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            """
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            """
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(
                device), dec_inputs.to(device), dec_outputs.to(device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(
                enc_inputs, dec_inputs)
            # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ==========================================================================================
    # 预测阶段
    # 测试集

    sentences = [
        # enc_input                dec_input           dec_output
        ['我 有  ', 'S', '']
    ]

    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    test_loader = Data.DataLoader(
        MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
    enc_inputs, _, _ = next(iter(test_loader))

    print()
    print("=" * 30)
    print("利用训练好的Transformer模型将中文句子'我 有 零 个 女 朋 友' 翻译成英文句子: ")
    for i in range(len(enc_inputs)):
        greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(
            1, -1).to(device), start_symbol=tgt_vocab["S"])
        print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
        print([src_idx2word[t.item()] for t in enc_inputs[i]], '->',
              [idx2word[n.item()] for n in greedy_dec_predict.squeeze()])


if __name__ == '__main__':
    main()