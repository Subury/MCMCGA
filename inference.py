import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score

import sys
sys.path.append('/workspace/protptype')
from pycocoevalcap.cider import Cider
from pycocoevalcap.rouge import Rouge
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import models
from utils.pretrain_datasets import MultimodalBertDataset
# from utils.datasets import Multi_Modal_Dataset as MultimodalBertDataset

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

CLINICALS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
             'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
             'Lung Opacity', 'Pleural Effusion', 'Pneumonia', 
             'Pneumothorax', 'Pleural Other', 'Support Devices', 'No Finding']

batch_size = 256
task = 'retrieval'

# multi-scans to same report
def scan_report_map(dataset):

    def get_key_tag(name):
        return "-".join(name.split('/')[1:3])
    
    unique_key_tags = set()
    report_to_scan_index_map = {}
    scan_to_report_index_map = {}
    for index, scan_name in enumerate(dataset.images_list):
        key_tag = get_key_tag(scan_name)
        
        if key_tag not in unique_key_tags:
            unique_key_tags.add(key_tag)
            report_to_scan_index_map[len(unique_key_tags)] = [index]
        else:
            report_to_scan_index_map[len(unique_key_tags)].append(index)
        
        scan_to_report_index_map[index] = len(unique_key_tags) - 1
    
    unique_report_index = [report_to_scan_index_map[key][-1] for key in sorted(list(report_to_scan_index_map.keys()))]

    return unique_report_index, scan_to_report_index_map, report_to_scan_index_map

def openi_scan_report_map(dataset):

    def get_key_tag(name):
        return "-".join(name.split('/')[-1].split('-')[:-1])
    
    unique_key_tags = set()
    report_to_scan_index_map = {}
    scan_to_report_index_map = {}
    for index, scan_name in enumerate(dataset.images_list):
        key_tag = get_key_tag(scan_name)
        
        if key_tag not in unique_key_tags:
            unique_key_tags.add(key_tag)
            report_to_scan_index_map[len(unique_key_tags)] = [index]
        else:
            report_to_scan_index_map[len(unique_key_tags)].append(index)
        
        scan_to_report_index_map[index] = len(unique_key_tags) - 1
    
    unique_report_index = [report_to_scan_index_map[key][-1] for key in sorted(list(report_to_scan_index_map.keys()))]

    return unique_report_index, scan_to_report_index_map, report_to_scan_index_map

def retrieval(scan_embeddings, report_embeddings, unique_report_index, 
              scan_to_report_index_map, report_to_scan_index_map):

    scan2report = {'Recall@1': 0, 'Recall@5': 0, 'Recall@10': 0, 'coarse_index': []}
    report2scan = {'Recall@1': 0, 'Recall@5': 0, 'Recall@10': 0, 'coarse_index': []}

    scan_embeddings = F.normalize(scan_embeddings, dim=-1, p=2).numpy()
    report_embeddings = F.normalize(report_embeddings, dim=-1, p=2).numpy()

    for index, scan_embedding in enumerate(scan_embeddings):

        scan_embedding = scan_embeddings[index, :]
        sim = 0.5 + 0.5 * np.dot(scan_embedding[np.newaxis, :], report_embeddings[unique_report_index, :].T)[0]
        sort_index = np.argsort(sim)[::-1]

        identity_labels = scan_to_report_index_map[index]

        for topk in [1, 5, 10]:

            if identity_labels in sort_index[:topk]:
                scan2report[f'Recall@{topk}'] += 1

        scan2report['coarse_index'].append(sort_index[:100])
    
    for topk in [1, 5, 10]:
        scan2report[f'Recall@{topk}'] = scan2report[f'Recall@{topk}'] / len(scan_embeddings) * 100
    
    print('scan to report: \n   ', 
          ", ".join([f"Recall@{topk}: {scan2report[f'Recall@{topk}']:.3f}%" for topk in [1, 5, 10]]))

    for index, report_embedding in enumerate(report_embeddings[unique_report_index, :]):

        report_embedding = report_embeddings[unique_report_index, :][index, :]
        sim = 0.5 + 0.5 * np.dot(report_embedding[np.newaxis, :], scan_embeddings.T)[0]
        sort_index = np.argsort(sim)[::-1]

        identity_labels = report_to_scan_index_map[index + 1]

        for topk in [1, 5, 10]:

            if len(np.intersect1d(np.array(identity_labels), sort_index[:topk])):

                if len(identity_labels) < topk:
                    report2scan[f'Recall@{topk}'] += len(np.intersect1d(np.array(identity_labels), sort_index[:topk])) / len(identity_labels)
                else:
                    report2scan[f'Recall@{topk}'] += len(np.intersect1d(np.array(identity_labels), sort_index[:topk])) / topk

        report2scan['coarse_index'].append(sort_index[:100])
    
    for topk in [1, 5, 10]:
        report2scan[f'Recall@{topk}'] = report2scan[f'Recall@{topk}'] / len(unique_report_index) * 100

    print('report to scan: \n   ', 
          ", ".join([f"Recall@{topk}: {report2scan[f'Recall@{topk}']:.3f}%" for topk in [1, 5, 10]]))

    return scan2report, report2scan

def multi_retrieval(scan_embeddings, report_embeddings, unique_report_index, 
              scan_to_report_index_map, report_to_scan_index_map):

    scan2report = {'Recall@1': 0, 'Recall@5': 0, 'Recall@10': 0, 'coarse_index': []}
    report2scan = {'Recall@1': 0, 'Recall@5': 0, 'Recall@10': 0, 'coarse_index': []}

    scan_embeddings = {key: F.normalize(scan_embeddings[key], dim=-1, p=2).numpy() for key in scan_embeddings.keys()}
    report_embeddings = {key: F.normalize(report_embeddings[key], dim=-1, p=2).numpy() for key in report_embeddings.keys()}

    for index in range(scan_embeddings[list(scan_embeddings.keys())[0]].shape[0]):

        scan_embedding = {key: scan_embeddings[key][index, :] for key in scan_embeddings.keys()}
        sim = {key: 0.5 + 0.5 * np.dot(scan_embedding[key][np.newaxis, :], report_embeddings[key][unique_report_index, :].T)[0] for key in scan_embedding.keys()}
        sort_index = np.argsort(np.sum(np.concatenate([sim[key][np.newaxis, :] for key in sim.keys()], axis=0), axis=0))[::-1]

        identity_labels = scan_to_report_index_map[index]

        for topk in [1, 5, 10]:

            if identity_labels in sort_index[:topk]:
                scan2report[f'Recall@{topk}'] += 1

        scan2report['coarse_index'].append(sort_index[:100])
    
    for topk in [1, 5, 10]:
        scan2report[f'Recall@{topk}'] = scan2report[f'Recall@{topk}'] / scan_embeddings[list(scan_embeddings.keys())[0]].shape[0] * 100
    
    print('scan to report: \n   ', 
          ", ".join([f"Recall@{topk}: {scan2report[f'Recall@{topk}']:.3f}%" for topk in [1, 5, 10]]))

    for index in range(len(unique_report_index)):

        report_embedding = {key: report_embeddings[key][unique_report_index, :][index, :] for key in report_embeddings.keys()}
        sim = {key: 0.5 + 0.5 * np.dot(report_embedding[key][np.newaxis, :], scan_embeddings[key].T)[0] for key in scan_embedding.keys()}
        sort_index = np.argsort(np.sum(np.concatenate([sim[key][np.newaxis, :] for key in sim.keys()], axis=0), axis=0))[::-1]

        identity_labels = report_to_scan_index_map[index + 1]

        for topk in [1, 5, 10]:

            if len(np.intersect1d(np.array(identity_labels), sort_index[:topk])):

                if len(identity_labels) < topk:
                    report2scan[f'Recall@{topk}'] += len(np.intersect1d(np.array(identity_labels), sort_index[:topk])) / len(identity_labels)
                else:
                    report2scan[f'Recall@{topk}'] += len(np.intersect1d(np.array(identity_labels), sort_index[:topk])) / topk

        report2scan['coarse_index'].append(sort_index[:100])
    
    for topk in [1, 5, 10]:
        report2scan[f'Recall@{topk}'] = report2scan[f'Recall@{topk}'] / len(unique_report_index) * 100

    print('report to scan: \n   ', 
          ", ".join([f"Recall@{topk}: {report2scan[f'Recall@{topk}']:.3f}%" for topk in [1, 5, 10]]))

    return scan2report, report2scan

transform_validation = transforms.Compose([
        transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4978], std=[0.2449])])

dataset_validation = MultimodalBertDataset('./dataset/mimic-cxr/official_protocol_test.csv', 
                                           '/dataset/mimic_cxr_ap-pa_dataset/files',
# dataset_validation = MultimodalBertDataset('/workspace/cxr-clip/openi_front_later.csv', 
                                        #    '/workspace/codeup/NLMCXR_dcm/official_PNG',
                                           transform=transform_validation,
                                           mask_prob=0.0,
                                           max_caption_length=512,
                                           token_name="/workspace/Bio_ClinicalBERT")

data_loader_validation = torch.utils.data.DataLoader(dataset_validation,
                                                     batch_size=batch_size,
                                                     num_workers=0,
                                                     pin_memory=False,
                                                     collate_fn=dataset_validation.collate_fn, 
                                                     shuffle=False)

unique_report_index, scan_to_report_index_map, report_to_scan_index_map = scan_report_map(dataset_validation)
# unique_report_index, scan_to_report_index_map, report_to_scan_index_map = openi_scan_report_map(dataset_validation)

model = models.__dict__['fdt'](norm_pix_loss=True)
for epoch in range(37, 38):

    checkpint_names = [f'/workspace/codeup/prototype/exp/exp-fdt-20240106111852/weights/checkpoint-0{epoch}.pth']
    print(checkpint_names[0])
    checkpint_weights = torch.load(checkpint_names[0], map_location='cpu')['model']
    for checkpint_name in checkpint_names[1:]:
        weights = torch.load(checkpint_name, map_location='cpu')['model']
        for key in weights.keys():
            checkpint_weights[key] += weights[key]

    for key in checkpint_weights.keys():
        checkpint_weights[key] /= len(checkpint_names)


    model.load_state_dict(checkpint_weights)
    model.to('cuda:0')
    model.eval()

    outputs_features = {}
    attentions = {'scan': torch.zeros((len(dataset_validation), 256), dtype=torch.float32),
                  'report': torch.zeros((len(dataset_validation), 256), dtype=torch.float32)}

    for key in ['global', 'prototype']:
        outputs_features[key] = {}
        outputs_features[key]['scan'] = torch.zeros((len(dataset_validation), 512), dtype=torch.float32)
        outputs_features[key]['report'] = torch.zeros((len(dataset_validation), 512), dtype=torch.float32)

    for batch in tqdm(data_loader_validation):

        with torch.no_grad():
            
            outputs = model(batch, mask_ratio=0.0, is_training=False)

            attentions['scan'][batch['index']] += outputs['attention']['scan'].cpu()
            attentions['report'][batch['index']] += outputs['attention']['report'].cpu()

            for key in outputs_features.keys():
                outputs_features[key]['scan'][batch['index']] += outputs[key]['scan'].cpu()
                outputs_features[key]['report'][batch['index']] += outputs[key]['report'].cpu()
    
    # import cv2
    print('Max value:', np.argmax((attentions['scan'] > 0).numpy().sum(axis=0)), attentions['scan'][:, 226].mean(), attentions['scan'][:, 226].max(), attentions['scan'][:, 226].min())
    # print('Max value:', (attentions['report'] > 0).numpy().sum(axis=0).max())
    # scan_depth = cv2.applyColorMap((attentions['scan'].reshape(-1, 16, 16).mean(dim=0).numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # cv2.imwrite('scan_prototype.jpg', scan_depth)
    # exit(0)

    for key in outputs_features.keys():
        scan2report, report2scan = retrieval(outputs_features[key]['scan'], outputs_features[key]['report'], 
                                            unique_report_index, scan_to_report_index_map, report_to_scan_index_map)
    
    multi_retrieval({key: outputs_features[key]['scan'] for key in outputs_features.keys()}, 
                    {key: outputs_features[key]['report'] for key in outputs_features.keys()},
                    unique_report_index, scan_to_report_index_map, report_to_scan_index_map)

    #     unique_report_list = [data_loader_validation.dataset.report_list[index] for index in unique_report_index]

    #     collect_results = {'BLEU_4': [], 'METEOR': [], 'ROUGE_L': [], 'CIDER': []}
    #     scorers = [
    #         (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
    #         (Meteor(), "METEOR"),
    #         (Rouge(), "ROUGE_L"),
    #         (Cider(), "CIDER")
    #     ]

    #     for topk in tqdm(range(1, 11)):
    #         eval_res = {}
    #         # Compute score for each metric
    #         for scorer, method in scorers:
    #             try:
    #                 score, scores = scorer.compute_score(
    #                                     {index: [" ".join(data_loader_validation.dataset.report_list[index])] 
    #                                         for index in range(len(scan2report['coarse_index']))}, 
    #                                     {index: [" ".join(unique_report_list[indexes[topk-1]])] 
    #                                         for index, indexes in enumerate(scan2report['coarse_index'])}, verbose=0)
    #             except TypeError:
    #                 score, scores = scorer.compute_score(
    #                                     {index: [" ".join(data_loader_validation.dataset.report_list[index])] 
    #                                         for index in range(len(scan2report['coarse_index']))}, 
    #                                     {index: [" ".join(unique_report_list[indexes[topk-1]])] 
    #                                         for index, indexes in enumerate(scan2report['coarse_index'])})
    #             if type(method) == list:
    #                 for sc, m in zip(score, method):
    #                     eval_res[m] = sc
    #             else:
    #                 eval_res[method] = score

    #         for key in collect_results.keys():
    #             collect_results[key].append(eval_res[key])
    #     print(collect_results)

    #     collect_results = {'BLEU_4': [], 'METEOR': [], 'ROUGE_L': [], 'CIDER': []}
    #     scorers = [
    #         (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
    #         (Meteor(), "METEOR"),
    #         (Rouge(), "ROUGE_L"),
    #         (Cider(), "CIDER")
    #     ]

    #     for topk in tqdm(range(1, 11)):
    #         eval_res = {}
    #         # Compute score for each metric
    #         for scorer, method in scorers:
    #             try:
    #                 score, scores = scorer.compute_score(
    #                                     {index: [" ".join(unique_report_list[index])] 
    #                                         for index in range(len(report2scan['coarse_index']))}, 
    #                                     {index: [" ".join(data_loader_validation.dataset.report_list[indexes[topk-1]])] 
    #                                         for index, indexes in enumerate(report2scan['coarse_index'])}, verbose=0)
    #             except TypeError:
    #                 score, scores = scorer.compute_score(
    #                                     {index: [" ".join(unique_report_list[index])] 
    #                                         for index in range(len(report2scan['coarse_index']))}, 
    #                                     {index: [" ".join(data_loader_validation.dataset.report_list[indexes[topk-1]])] 
    #                                         for index, indexes in enumerate(report2scan['coarse_index'])})
    #             if type(method) == list:
    #                 for sc, m in zip(score, method):
    #                     eval_res[m] = sc
    #             else:
    #                 eval_res[method] = score

    #         for key in collect_results.keys():
    #             collect_results[key].append(eval_res[key])
                
    #     print(collect_results)