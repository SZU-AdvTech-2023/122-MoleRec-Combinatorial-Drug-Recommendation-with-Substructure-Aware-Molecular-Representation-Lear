import pickle

import pandas as pd
import dill
import numpy as np
from collections import defaultdict

import torch
from rdkit import Chem
from rdkit.Chem import Recap
from rdkit.Chem import BRICS


##### process medications #####
# load med data
# 只保留PRESCRIPTIONS文件中的SUBJECT_ID  HADM_ID  STARTDATE  ENDDATE  NDC这些内容
# 并进行去重，去处NDC为零的数据，整理顺序等
def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'})

    med_pd.drop(columns=[
        'ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
        'FORMULARY_DRUG_CD', 'PROD_STRENGTH', 'DOSE_VAL_RX',
        'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'GSN',
        'FORM_UNIT_DISP', 'ROUTE', 'ENDDATE', 'DRUG'
    ], axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(
        med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S'
    )
    med_pd.sort_values(
        by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True
    )
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd

# medication mapping

# 把NDC号通过ndc2rxnorm_mapping.txt文件，转换成RXCUI号，存储在新的列中
# 把NDC转换成ACT号，
def ndc2atc4(med_pd):
    with open(ndc_rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(
        index=med_pd[med_pd['RXCUI'].isin([''])].index,
        axis=0, inplace=True
    )

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC5': 'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd

# visit >= 2

# 去除HADM_ID长度小于2的数据，
# TODO：可能是为了去除异常的数据
def process_visit_lg2(med_pd):
    a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(
        by='SUBJECT_ID'
    )['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
    a = a[a['HADM_ID_Len'] > 1]
    return a

# most common medications

# 筛选NDC（即ACT）出现频率最高的前300种药物，其他忽略掉
def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().\
        rename(columns={0: 'count'}).\
        sort_values(by=['count'], ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]

    return med_pd.reset_index(drop=True)

##### process diagnosis #####

# 删除SEQ_NUM  ROW_ID， 并筛选出频率最高的前2000的ICD9的数据
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM', 'ROW_ID'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().\
            reset_index().rename(columns={0: 'count'}).\
            sort_values(by=['count'], ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(
            diag_count.loc[:1999, 'ICD9_CODE']
        )]

        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd

##### process procedure #####


def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE': 'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().\
        rename(columns={0: 'count'}).\
        sort_values(by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(
        pro_count.loc[:1000, 'ICD9_CODE']
    )]

    return pro_pd.reset_index(drop=True)

###### combine three tables #####


def combine_process(med_pd, diag_pd, pro_pd):

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(
        diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(
        pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(
        combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(
        combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(
        combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].\
        unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].\
        unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].\
        unique().reset_index().rename(columns={'ICD9_CODE': 'PRO_CODE'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    # data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    # 最后data中包含：[SUBJECT_ID, HADM_ID, ICD9_CODE, NDC, PRO_CODE, NDC_Len]
    return data


def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))

    diag = data['ICD9_CODE'].values
    med = data['NDC'].values
    pro = data['PRO_CODE'].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))

    avg_diag, avg_med, avg_pro, max_diag, max_med, max_pro, \
        cnt, max_visit, avg_visit = [0 for i in range(9)]

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['NDC']))
            z.extend(list(row['PRO_CODE']))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print('#avg of diagnoses ', avg_diag / cnt)
    print('#avg of medicines ', avg_med / cnt)
    print('#avg of procedures ', avg_pro / cnt)
    print('#avg of vists ', avg_visit / len(data['SUBJECT_ID'].unique()))

    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)

# indexing file and final record


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

# create voc set

# 创建 id <--> str 之间的映射，就是把诊断、操作、药物编码当作字符串，并给其赋予一个id
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['NDC'])
        pro_voc.add_sentence(row['PRO_CODE'])

    dill.dump(obj={'diag_voc': diag_voc, 'med_voc': med_voc,
                   'pro_voc': pro_voc}, file=open('voc_final.pkl', 'wb'))
    return diag_voc, med_voc, pro_voc

# create final records

# 形成病人记录，一个病人多次就诊记录，每次就诊记录包括ICD9_CODE PRO_CODE NDC三项数据
# 例如：[维度1病人
#       [维度2就诊次序
#        [[0, 1, 2, 3, 4, 5, 6, 7], 第一次就诊的ICD9_CODE
#         [0, 1, 2],第一次就诊的PRO_CODE
#         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], 第一次就诊的NDC
#        [[8, 9, 10, 7],第二次就诊的ICD9_CODE
#         [3, 4, 1],第二次就诊的PRO_CODE
#         [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17]]]]第二次就诊的NDC
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = []  # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['NDC']])
            patient.append(admission)
        records.append(patient)
    dill.dump(obj=records, file=open('records_final.pkl', 'wb'))
    return records


# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):

    TOPK = 40  # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)

    with open(cid_atc, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])

    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect
    ddi_most_pd = \
        ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).\
        size().reset_index().rename(columns={0: 'count'}).\
        sort_values(by=['count'], ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:, :]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(
        ddi_most_pd[['Side Effect Name']],
        how='inner', on=['Side Effect Name']
    )
    ddi_df = fliter_ddi_df[['STITCH 1', 'STITCH 2']].\
        drop_duplicates().reset_index(drop=True)

    # weighted ehr adj
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j <= i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
    dill.dump(ehr_adj, open('ehr_adj_final.pkl', 'wb'))

    # ddi adj
    ddi_adj = np.zeros((med_voc_size, med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row['STITCH 1']
        cid2 = row['STITCH 2']

        # cid -> atc_level3
        # 化合物CID 对应的ATC列表
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:

                # atc_level3 -> atc_level4
                # 从化合物对应的ATC列表取出ATC编号
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        # 如果两种化合物具有不同
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i],
                                    med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j],
                                    med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open('ddi_A_final.pkl', 'wb'))

    return ddi_adj

def decomposed_med_structure(molecule, med_voc, decomposer='BRICS'):
    result_smile = []
    print(len(med_voc.items()))  # 131
    for index, ndc in med_voc.items():
        smilesList = list(molecule[ndc])
        for smiles in smilesList:
            # todo: 消融实验：BRICS-> RECAP
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                if decomposer == 'RECAP':
                    decomposer_mols = Recap.RecapDecompose(mol)
                    substructrue = decomposer_mols.GetLeaves().keys()
                elif decomposer == 'BRICS':
                    substructrue = BRICS.BRICSDecompose(mol)
                else:
                    raise ValueError("No such decomposer")
                for fragment in substructrue:
                    if fragment not in result_smile:
                        result_smile.append(fragment)
    return result_smile


if __name__ == '__main__':
    # please change into your own MIMIC folder
    med_file = 'PRESCRIPTIONS.csv'
    diag_file = 'DIAGNOSES_ICD.csv'
    procedure_file = 'PROCEDURES_ICD.csv'

    med_structure_file = './idx2SMILES.pkl'

    # drug code mapping files
    ndc2atc_file = './ndc2atc_level4.csv'
    cid_atc = './drug-atc.csv'
    ndc_rxnorm_file = './ndc2rxnorm_mapping.txt'

    # ddi information
    ddi_file = './drug-DDI.csv'

    # for med

    # 只保留PRESCRIPTIONS文件中的SUBJECT_ID  HADM_ID  STARTDATE  ENDDATE  NDC这些内容
    # 并进行去重，去处NDC为零的数据，整理顺序等
    med_pd = med_process(med_file)
    # 去除异常数据
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    # 融合，只保留med_pd中和med_pd_lg2共有的数据
    # 还是去除异常数据的处理
    med_pd = med_pd.merge(
        med_pd_lg2[['SUBJECT_ID']],
        on='SUBJECT_ID', how='inner'
    ).reset_index(drop=True)
    # NDC号转换成ACT号
    med_pd = ndc2atc4(med_pd)
    # 去除不存在ACT <-> SMILES 配对的数据
    NDCList = dill.load(open(med_structure_file, 'rb'))
    med_pd = med_pd[med_pd.NDC.isin(list(NDCList.keys()))]
    # 筛选出NDC出现频率最高的前300种药品
    med_pd = filter_300_most_med(med_pd)

    # 最后med_pd包含：
    # SUBJECT_ID  HADM_ID  STARTDATE  ENDDATE  NDC 这几项数据
    # 首先NDC已经替换成ATC，然后且对数据进行了处理，去除了异常的数据（存在内容缺失、匹配缺失等），最后筛选出频率前300的数据

    print('complete medication processing')

    # for diagnosis(诊断)
    # 删除SEQ_NUM ROW_ID， 并筛选出频率最高的前2000的ICD9的数据+
    diag_pd = diag_process(diag_file)

    print('complete diagnosis processing')

    # for procedure
    # 删除SEQ_NUM ROW_ID，
    pro_pd = procedure_process(procedure_file)
    # pro_pd = filter_1000_most_pro(pro_pd)

    print('complete procedure processing')

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)
    statistics(data)
    # # 最后data中包含：[SUBJECT_ID, HADM_ID, ICD9_CODE, NDC, PRO_CODE, NDC_Len]
    data.to_pickle('data_final.pkl')
    print('complete combining')

    # ddi_matrix
    # 获得诊断、药品、操作的词汇表
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    # 构建每位病人的记录，每位病人的记录包含：多次就诊记录，每次就诊记录又包括ICD9_CODE PRO_CODE NDC三项数据
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)
    #
    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)

    print('ok')
    # 把完整的药物分解为子结构
    # todo:消融实验，BRICS 改为 RECAP方法
    # 子结构的处理在ddi_mask_H.py文件中
    # molecule = dill.load(open(med_structure_file, 'rb'))
    # substructure_smiles = decomposed_med_structure(molecule, med_voc.idx2word, decomposer='RECAP')
    # with open('substructure_smiles.pkl', 'wb') as f:
    #     pickle.dump(substructure_smiles, f)
    # print('complete decomposed')

