# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model

from wenet.cif.search.beam_search import build_beam_search
import torch
import torchaudio

import os





def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring',
                            'rnnt_greedy_search', 'rnnt_beam_search',
                            'rnnt_beam_attn_rescoring',
                            'ctc_beam_td_attn_rescoring', 'hlg_onebest',
                            'hlg_rescore', 'cif_greedy_search',
                            'cif_beam_search',
                        ],
                        default='attention',
                        help='decoding mode')

    parser.add_argument('--search_ctc_weight',
                        type=float,
                        default=1.0,
                        help='ctc weight for nbest generation')
    parser.add_argument('--search_transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for nbest generation')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode')

    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                             'transducer attention rescore mode')
    parser.add_argument('--attn_weight',
                        type=float,
                        default=0.0,
                        help='attention weight for rescoring weight in '
                             'transducer attention rescore mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument('--connect_symbol',
                        default='',
                        type=str,
                        help='used to connect the output characters')

    parser.add_argument('--word',
                        default='',
                        type=str,
                        help='word file, only used for hlg decode')
    parser.add_argument('--hlg',
                        default='',
                        type=str,
                        help='hlg file, only used for hlg decode')
    parser.add_argument('--lm_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--r_decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')

    args = parser.parse_args()
    print(args)
    return args

import torch
import matplotlib.pyplot as plt
import numpy as np




def get_labformat(path, subsample,char_dict,transcript):
    index=path[0][0]
    begin = path[0][1]*0.01*subsample
    duration = 0
    labformat = []
    count=0
    for point in (path):

        # print(point)
        if point[0]>index:
            duration=count*0.01*subsample
            print("{:.2f} {:.2f} {}".format(begin, begin + duration,
                                            transcript[point[0]-1]))
            labformat.append("{:.2f} {:.2f} {}\n".format(
                begin, begin + duration, transcript[point[0]-1]))



            index=point[0]
            begin=point[1]*0.01*subsample
            count=1
        else:
            count+=1

    duration=count*0.01*subsample
    print("{:.2f} {:.2f} {}".format(begin, begin + duration,
                                            transcript[-1]))
    labformat.append("{:.2f} {:.2f} {}\n".format(
                begin, begin + duration, transcript[-1]))
    
        
        # # time duration
        # duration = len(t) * 0.01 * subsample
        # if idx < len(timestamp) - 1:
        #     print("{:.2f} {:.2f} {}".format(begin, begin + duration,
        #                                     char_dict[t[-1]]))
        #     labformat.append("{:.2f} {:.2f} {}\n".format(
        #         begin, begin + duration, char_dict[t[-1]]))
        # else:
        #     non_blank = 0
        #     for i in t:
        #         if i != 0:
        #             token = i
        #             break
        #     print("{:.2f} {:.2f} {}".format(begin, begin + duration,
        #                                     char_dict[token]))
        #     labformat.append("{:.2f} {:.2f} {}\n".format(
        #         begin, begin + duration, char_dict[token]))
        # # begin = begin + duration
    return labformat



def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring',
                     'cif_beam_search', ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           non_lang_syms,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model = init_model(configs)

    # Load dict
    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    model.eval()

    # Build BeamSearchCIF object
    if args.mode == 'cif_beam_search':
        cif_beam_search = build_beam_search(model, args, device)
    else:
        cif_beam_search = None
    
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            # 对于batchsize=1的输入而言，【batchsize,len,80】80为输入的每一帧的维度大小
            # 如果batchsize》1那么输入就为【batchsize,max-frame-len,80】加上feats-length表示每一个的frame大小，更少frame数量的音频结尾部分为0
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
           
            file_name='/home/chenyang/chenyang_space/data/aishell_test_conformer_attention/attention_matrix.txt'
            with open(file_name, 'a') as file:
                file.write(str(keys)+'\n')
            if args.mode == 'attention':
                hyps, _ = model.align_attention(
                    target,
                    target_lengths,
                    feats,
                    feats_lengths,
                    beam_size=args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp.tolist() for hyp in hyps]

            for i, key in enumerate(keys):
                content = []
                for w in hyps[i]:
                    if w == eos:
                        break
                    content.append(char_dict[w])
                logging.info('{} {}'.format(key, args.connect_symbol
                                            .join(content)))
                fout.write('{} {}\n'.format(key, args.connect_symbol
                                            .join(content)))


from textgrid import TextGrid, IntervalTier
def generator_textgrid(maxtime, lines, output):
    # Download Praat: https://www.fon.hum.uva.nl/praat/
    interval = maxtime / (len(lines) + 1)
    margin = 0.0001

    tg = TextGrid(maxTime=maxtime)
    linetier = IntervalTier(name="line", maxTime=maxtime)

    i = 0
    for l in lines:
        s, e, w = l.split()
        linetier.add(minTime=float(s) + margin, maxTime=float(e), mark=w)

    tg.append(linetier)
    # print("successfully generator {}".format(output))
    tg.write(output)

def align_test_set():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring',
                     'cif_beam_search', ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           non_lang_syms,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model = init_model(configs)

    # Load dict
    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    model.eval()
    diction={}
    with open('/home/chenyang/chenyang_space/wenet/wenet/examples/aishell/s0/data/dict/lang_char.txt','r') as file:
        for line in file:
            # print(line.split())
            diction[line.split()[0]]=int(line.split()[1])

    base=0
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            # 对于batchsize=1的输入而言，【batchsize,len,80】80为输入的每一帧的维度大小
            # 如果batchsize》1那么输入就为【batchsize,max-frame-len,80】加上feats-length表示每一个的frame大小，更少frame数量的音频结尾部分为0
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            print('keys')
            print(keys)
 
            base=align_one_file(keys, feats, target, feats_lengths, target_lengths,diction,base)
            

           
            # file_name='/home/chenyang/chenyang_space/data/aishell_test_conformer_attention/attention_matrix.txt'
            # with open(file_name, 'a') as file:
            #     file.write(str(keys)+'\n')
            # if args.mode == 'attention':
            #     hyps, _ = model.align_attention(
            #         target,
            #         target_lengths,
            #         feats,
            #         feats_lengths,
            #         beam_size=args.beam_size,
            #         decoding_chunk_size=args.decoding_chunk_size,
            #         num_decoding_left_chunks=args.num_decoding_left_chunks,
            #         simulate_streaming=args.simulate_streaming)
            #     hyps = [hyp.tolist() for hyp in hyps]

            # for i, key in enumerate(keys):
            #     content = []
            #     for w in hyps[i]:
            #         if w == eos:
            #             break
            #         content.append(char_dict[w])
            #     logging.info('{} {}'.format(key, args.connect_symbol
            #                                 .join(content)))
            #     fout.write('{} {}\n'.format(key, args.connect_symbol
            #                                 .join(content)))
    return None
def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    # 如果未找到匹配的键，则返回None或抛出异常
    return None
def align_one_file(keys, feats, target, feats_lengths, target_lengths,diction,base):
    transcription=''
    # print(diction)
    for i in range(target_lengths[0]):
        transcription+=find_key_by_value(diction,target[0][i])
    # print(transcription)

    
    file_tensor=None
    
    for i in range(target_lengths[0]+2):
        index=int(base)+i*12+11
        file_base='/home/chenyang/chenyang_space/data/aishell_test_conformer_attention/tensors/tensor_'
        loaded_tensor = torch.load(file_base+str(index)+'.pt')
        # print(loaded_tensor.shape)
        mean_tensor=torch.max(loaded_tensor,dim=1)

      
        if i==0:
            file_tensor=mean_tensor[0]
        else:
            file_tensor=torch.cat((file_tensor,mean_tensor[0]),dim=1)
    file_tensor=torch.squeeze(file_tensor)
    print(file_tensor.shape)
  
    # 创建一个张量
    

    # 将张量转换为 NumPy 数组
    array = file_tensor.cpu().numpy()
    array = np.delete(array, len(array)-1, axis=0)
    array = np.delete(array, len(array)-1, axis=0)

    # 绘制热度图
    # plt.imshow(array, cmap='gray', interpolation='nearest')
    # # plt.colorbar()

    # # 显示图形
    # plt.savefig('/home/chenyang/chenyang_space/data/aishell_test_conformer_attention/'+keys[0]+'.png')
#     dif=[]
#     for i in range(len(array)):
#         row_list=[]
#         for j in range(len(array[0])-1):
#             row_list.append((array[i][j]-array[i][j+1]))
#         dif.append(row_list)
#     # print('array0')
#     # print(array[0])
#     # print('dif0')
#     # print(dif[0])
#     # print('dif-1')
#     # print(dif[-1])
#     # start_index=np.argmin(dif[0])
#     end_index=np.argmax(dif[len(dif)-1])
# # 接下来搞定startindex
#     var_list=[]
#     # print(len(array))
#     for j in range(len(array[0])//2):
#         # print(array[:,j])
#         var_list.append(np.var(array[:,j]))
#     print(var_list)
#     flag=False
#     for i in range(len(var_list)):
#         if var_list[i]<0.00001:
#             flag=True

#         if flag==True and var_list[i]>0.001:
#             start_index=i
#             break








#     start_index+=1
    # 左闭合右闭合的
    # print(start_index)
    # print(end_index)
    start_index=1
    end_index=len(array[0])-1
    # 定义二维数组表示概率
    probabilities = array

    # 获取二维数组的形状
    rows, cols = probabilities.shape





    # 动态规划也失败傻逼玩意
    # 定义起始位置和目标位置
    start = (0, start_index)
    goal = (rows - 1, end_index)

    # 定义路径列表和概率值
    path = [start]
    path_prob = probabilities[start]

    # 创建一个与输入数组相同大小的二维数组，用于存储最大概率
    max_prob = np.zeros_like(probabilities)

    # 初始化起始位置的最大概率
    max_prob[0, start_index] = probabilities[0, start_index]

    # 计算每个位置的最大概率
    for i in range(1, rows):
        max_prob[i, start_index] =  probabilities[i, start_index]

    for j in range(start_index, end_index+1):
        max_prob[0, j] = max_prob[0, j-1] + probabilities[0, j]

    for i in range(1, rows):
        for j in range(start_index, end_index+1):
            max_prob[i, j] = max(max_prob[i-1, j-1], max_prob[i, j-1]) +probabilities[i, j]

    # 最大概率即为右下角位置的值
    max_probability = max_prob[rows-1, end_index]

    # 回溯路径
    path = [(rows-1, end_index)]
    i, j = rows-1, end_index
    while i > 0 or j > start_index:
        if i > 0 and max_prob[i-1, j-1] >= max_prob[i, j-1]:
            path.append((i-1, j-1))
            i -= 1
            j -=1
        else:
            path.append((i, j-1))
            j -= 1
    path.reverse()


    # # 开始贪婪搜索
    # while path[-1] != goal :
    #     current_row, current_col = path[-1]
        

    #     # 获取当前位置的邻居
    #     if current_row==rows-1:
    #         neighbors =[(current_row, current_col + 1)]
    #     else:
    #         if current_col==cols-1:
    #             print('error 提前到达了最后一帧，但是字还没有分配完')
    #         else:
    #             neighbors = [(current_row + 1, current_col+1), (current_row, current_col + 1)]


        
        

    #     # 计算邻居位置的概率值
    #     neighbor_probs = [probabilities[row, col] for row, col in neighbors]

    #     # 选择具有最大概率值的邻居位置
    #     max_index = np.argmax(neighbor_probs)
    #     next_row, next_col = neighbors[max_index]
    #     if next_row>rows-1:
    #         break
    #     if next_col>cols-1:
    #         break
    #     # 将下一个位置添加到路径中，并更新路径的概率值
    #     path.append((next_row, next_col))
    #     path_prob *= probabilities[next_row, next_col]

# 贪心算法效果一般
    


    # 打印找到的路径和对应的概率
    print("Path:", path)

    # 接下来按照帧来进行align生成文件
    # 先找到target
    # 先导入dict
    
    # print(diction)

    labformat=get_labformat(path,4,diction,transcription)
    # timestamp = get_frames_timestamp(alignment)
    result_path='/home/chenyang/chenyang_space/data/aishell_test_conformer_attention/'
    file_name=keys[0]
    textgrid_path = result_path+(file_name)+".TextGrid"
    generator_textgrid(maxtime=(path[-1][1] + 1) * 0.01 *
                            4,
                            lines=labformat,
                            output=textgrid_path)
    
    base=base+12*(target_lengths[0]+2)
    return base

            
  


    


    


if __name__ == '__main__':
    # main()
    # 接下来读取所有的attention数据，整理一下，画出图来
    # 先尝试搞一个的
    align_test_set()






 



