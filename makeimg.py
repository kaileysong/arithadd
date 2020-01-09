# -*- coding:utf-8 -*-
# author: zzh time: 2019/10/18
from PIL import Image, ImageDraw, ImageFont
import os
import random


def draw_image(new_img, text, show_image=False):

    draw = ImageDraw.Draw(new_img)
    img_size = new_img.size

    font_size = 60
    fnt = ImageFont.truetype('Sim.ttf', font_size)
    fnt_size = fnt.getsize(text)
    while fnt_size[0] > img_size[0] or fnt_size[0] > img_size[0]:
        font_size -= 5
        fnt = ImageFont.truetype('Sim.ttf', font_size)
        fnt_size = fnt.getsize(text)

    x = (img_size[0] - fnt_size[0]) / 2
    y = (img_size[1] - fnt_size[1]) / 2 -15
    draw.text((x, y), text, font=fnt, fill=(255))

def new_image(width, height, text='default', color=(0), show_image=False):
    new_img = Image.new('L', (int(width), int(height)), color)
    draw_image(new_img, text, show_image)
    new_img.save(r'%s.png' % (text))
    del new_img


def new_image_with_file(fn):
    with open(fn, encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if l:
                ls = l.split(',')
                if '#' == l[0] or len(ls) < 2:
                    continue

                new_image(*ls)


if '__main__' == __name__:
    f = 0
    for i in range(100):
        l = 99
        for j in range(100):

           new_image(224, 224, str(f)+'+'+str(l), show_image=True)
           label = f + l
           dir = os.getcwd() + '/'
           alldata = open(os.getcwd() + '/' + 'alldata.txt', 'a')
           if f == l:
               name = 'equal' + ' ' + str(dir) + str(f) + '+' + str(l) + '.png' + ' ' + str(int(label)) + '\n'
               alldata.write(name)
           if f < l:
               name = str(f) + str(l) + ' ' + str(dir) + str(f) + '+' + str(l) + '.png' + ' ' + str(int(label)) + '\n'
               alldata.write(name)
           if f > l:
               name = str(l) + str(f) + ' ' + str(dir) + str(f) + '+' + str(l) + '.png' + ' ' + str(int(label)) + '\n'
               alldata.write(name)
           alldata.close()
           l = l - 1
        f = f + 1

