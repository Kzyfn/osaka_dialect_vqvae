import glob
import os
from os.path import expanduser, join

jl_dir = './data/jl_out_lab' #Juliusから出力されたアラインメント済みラベルが格納してある場所
ojt_dir = './ojt_lab' #OpenJTalkから出力されたフルコンテキストラベルが格納してある場所
out_dir = './data/lab' #アラインメント済みフルコンテキストラベルが出力される場所

seg_labs = sorted(glob.glob(join(jl_dir, "*.lab")))
ojt_labs = sorted(glob.glob(join(ojt_dir, "*.lab")))


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for s, o in zip(seg_labs, ojt_labs):
    with open(s) as f_s:
        s_line = f_s.readlines()
    with open(o) as f_o:
        o_line = f_o.readlines()
        
    #print(len(s_line), len(o_line))
    #if len(s_line) != len(o_line) -2:
    print(s, o)
    out=''
    for itr, ol in enumerate(o_line):
        if itr == 0 or itr == len(o_line) - 1:
            continue
        ol = ol.split(' ')
        sl = s_line[itr-1].split(' ')
        #時刻の表記も違う
        ol[0] = str(int(10000000*float(sl[0])))
        ol[1] = str(int(10000000*float(sl[1])))

        out = out+' '.join(ol)

        with open(join(out_dir, os.path.basename(s)), mode='w') as o:
            o.write(out)
