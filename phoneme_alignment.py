import subprocess
import numpy as np
import glob
import os
from os.path import expanduser, join
import shutil
import subprocess

#各ツール・ファイルのディレクトリを指定してください
tools_dir = '../tts_tools'
openjtalk_path = '/usr/local/bin/open_jtalk'
dic_path = '/usr/local/share/open_jtalk_dic_utf_8-1.07/'
htsvoice_path = '/usr/local/share/hts_voice_nitech_jp_atr503_m001-1.05/nitech_jp_atr503_m001.htsvoice'
julius_path = './julius-alignment/segment_julius_with_monophone.pl'

corpus_dir = './data'
base_path = join(corpus_dir, 'base.scp')
wav_dir = join(corpus_dir, 'wav')
wav16_dir = join(wav_dir, '16k')

ojtlab_dir = './ojtlab'
lab_dir = './lab'
jl_in_dir = './jl_in_lab'
jl_out_dir = './jl_out_lab'


def divide_transcript_utf8(corpus_dir):
    with open(join(corpus_dir, 'transcript_utf8.txt'), mode='r') as f:
        lines = f.readlines()
    scp = ''
    out = []
    for line in lines:
        if line != '\n':
            a=line.split(':')
            out.append(a[1])
        scp = scp+a[0]+'\n'
    
    return out, scp

def openjtalk(openjtalk_path, dic_path, htsvoice_path, txt, name, ojtlab_dir):
    open_jtalk = [openjtalk_path]
    dic = ['-x', dic_path]
    htsvoice=['-m', htsvoice_path]
    ojtlab = ['-ot', join(ojtlab_dir, name)]
    
    cmd = open_jtalk + dic + htsvoice + ojtlab
    c = subprocess.Popen(cmd,stdin=subprocess.PIPE)
    c.stdin.write(txt.encode('utf-8'))
    c.stdin.close()

def ojtlab2monolab(ojtlab):
    lines = ojtlab.split('[Output label]')[1].split('[Global parameter]')[0].split('\n')
    lines = [x for x in lines if x]
    
    out = []
    for l in line:
        l = l.split(' ')[2]
        l = l.split('-')[1]
        l = l.split('+')[0]
        if l == 'sil':
            if i == 0:
                l = 'silB'
        if l == 'pau':
            l = 'sp'
        out.append(l+"\n")
    else:
        out[-1] = 'silE'

    return out

def downsampling(wav_dir, wav16_dir):
    wav16_dir = join(wav_dir, '16k')
    wav = glob.glob(join(wav_dir, '*.wav'))

    for w in wav:
        p = subprocess.Popen("sox {} -c 1 -b 16 -r 16000 -e signed-integer {}".format(w, join(wav16_dir, os.path.basename(w))),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

def fourced_align(julius_path, base_path, wav16_dir, jl_in_dir, jl_out_dir):
	c = subprocess.Popen("perl {} {} {} {} {}".format(julius_path, base_path, wav16_dir, jl_in_dir, jl_out_dir), shell=True)
	c.communicate()

def replace_with_aligned_time(ojtlab, jllab):
    oj_lines = ojtlab.split('[Output label]')[1].split('[Global parameter]')[0].split('\n')
    oj_lines = [x for x in oj_lines if x]
    jl_lines = jllab.split('\n')
    jl_lines = [x for x in jl_lines if x]
    out=[]
    for ol, jl in zip(oj_lines, jl_lines):
        ol = ol.split(' ')
        jl = jl.split(' ')
        
        ol[0] = str(int(10000000*float(jl[0])))
        ol[1] = str(int(10000000*float(jl[1])))

        out.append(' '.join(ol)+'\n')
    else:
        out[-1] = out[-1].rstrip('\n')

    return out

if __name__ == '__main__':
    if not os.path.exists(wav16_dir):
        os.makedirs(wav16_dir)
    if not os.path.exists(ojtlab_dir):
        os.makedirs(ojtlab_dir)
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    if not os.path.exists(jl_in_dir):
        os.makedirs(jl_in_dir)
    if not os.path.exists(jl_out_dir):
        os.makedirs(jl_out_dir)

    # #transcript_utf8.txtを1文ずつ分割, base.scpファイル生成
    texts, scp = divide_transcript_utf8(corpus_dir)
    with open(base_path, mode='w') as f:
        f.write(scp)

    # #OpenJTalkでフルコンテキストラベル生成
    for t, name in zip(texts, scp.split('\n')):
        openjtalk(openjtalk_path, dic_path, htsvoice_path, t, name+'.ojtlab', ojtlab_dir)

    #julius用モノフォンラベル生成
    ojtlab_paths = glob.glob(join(ojtlab_dir, "*.ojtlab"))
    for o in ojtlab_paths:
        with open(o, mode='r') as f:
            ojl = f.read()
        with open(join(jl_in_dir, os.path.basename(o).replace('ojtlab','lab')), mode='w') as f:
            f.writelines(ojtlab2monolab(ojl))
    
    #juliusを使うため16kHzにダウンサンプリング
    downsampling(wav_dir, wav16_dir)

    # #強制アラインメント
    fourced_align(julius_path, base_path, wav16_dir, jl_in_dir, jl_out_dir)

    #アラインメントされた時刻に差し替え
    jl_out_paths = glob.glob(join(jl_out_dir, '*.lab'))
    for ojtlab_path, jl_out_path in zip(ojtlab_paths, jl_out_paths):
        with open(ojtlab_path, mode='r') as f:
            ojt_fulllab = f.read()
        with open(jl_out_path, mode='r') as f:
            jl_monolab = f.read()
        
        with open(join(lab_dir, os.path.basename(jl_out_path)), mode='w') as f:
            f.writelines(replace_with_aligned_time(ojt_fulllab, jl_monolab))
