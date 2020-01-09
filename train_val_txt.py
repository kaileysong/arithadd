import os
import random


def train_valtxt():
    root = os.getcwd()
    f = open(os.path.join(root, 'alldata.txt'))
    all_lines = f.readlines()
    all_lines = set(all_lines)

    for line in all_lines:
        b = line.split(" ", 1)[0]
        if b == 'equal':
            val = open(os.path.join(root, 'val.txt'), 'a')
            val.write(line.split(" ", 1)[1])
            val.close()
            txtline = {line}
            txtline = set(txtline)
            all_lines = all_lines - txtline
    for i in range(4950):
        onetxt = random.sample(all_lines, 1)
        val = open(os.path.join(root, 'val.txt'), 'a')
        val.write(onetxt[0].split(" ", 1)[1])
        val.close()
        a = onetxt[0].split(" ", 1)[0]
        onetxt = set(onetxt)

        all_lines = all_lines - onetxt

        for line in all_lines:
            b = line.split(" ", 1)[0]
            if a == b:
                train = open(os.path.join(root, 'train.txt'), 'a')
                train.write(line.split(" ", 1)[1])
                train.close()
                txtline = {line}
                txtline = set(txtline)
                all_lines = all_lines - txtline

    print(len(all_lines))
    for file in all_lines:
        train = open(os.path.join(root, 'train.txt'), 'a')
        train.write(file.split(" ", 1)[1])
        train.close()

