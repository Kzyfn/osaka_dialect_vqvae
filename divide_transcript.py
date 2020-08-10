import fileinput
import os
from os.path import join

out_dir = './text'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

scp = ''
for i, line in enumerate(fileinput.input()):
    a=line.split(':')
    print(join(out_dir, a[0]+'.txt'))
    with open(join(out_dir, a[0]+'.txt'), mode='w') as f:
        f.write(a[1][a[1].find('\t')+1:])
    
    scp = scp+a[0]+'\n'

with open('./base.scp', mode='w') as f:
        f.write(scp)
