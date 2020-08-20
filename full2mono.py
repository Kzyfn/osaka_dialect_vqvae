import glob
import os
from os.path import expanduser, join

in_dir = './ojt_lab'
out_dir = './jl_in_lab'

ojt_labs = glob.glob(join(in_dir, "*.lab"))

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for i in ojt_labs:
    with open(i) as f:
        line = f.readlines()  
        out = []
        for j, l in enumerate(line):
            l = l.split(' ')[2]
            l = l.split('-')[1]
            l = l.split('+')[0]
            
			#OpenJTalkとJuliusは無音区間の表記が異なるので変更
            if l == 'sil':
                if j == 0:
                    l = 'silB'
                else:
                    l = 'silE'
            if l == 'pau':
                l = 'sp'
            
            out.append(l+"\n")

        with open(join(out_dir, os.path.basename(i)), mode='w') as o:
            o.writelines(out)
