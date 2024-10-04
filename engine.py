import wandb
import torch
import numpy as np

import utils.misc as misc
import utils.lr_sched as lr_sched

def train_one_epoch(model, data_loader, optimizer, device, epoch, 
                    loss_scaler, log_writer=None, args=None):
        
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    mask_ratio = args.mask_ratio
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header), start=1):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(batch, mask_ratio=mask_ratio, is_training=True, loss_keys=list(args.loss.keys()))
            loss_value = {key: loss[key].item() for key in loss.keys()} 

            loss = sum([args.loss[key] * loss[key] for key in args.loss.keys()])
            
            loss = loss / accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            
            if data_iter_step % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(**loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            loss_value_reduce = {key: misc.all_reduce_mean(loss_value[key]) for key in loss_value.keys()}
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and (data_iter_step + 1) % 100:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(((data_iter_step + 1) / len(data_loader) + epoch) * 1000)

                for key in loss_value_reduce.keys():
                    log_writer.add_scalar(f'train/{key}_loss', loss_value_reduce[key], epoch_1000x)

                log_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def retrieval(scan_embeddings, report_embeddings, image_identity_labels, 
              report_identity_indexes, indexes, device='cpu'):

    img2txt = {'topK(1)': torch.zeros([]).to(device), 
               'topK(5)': torch.zeros([]).to(device), 
               'topK(10)': torch.zeros([]).to(device)}
    
    for index in indexes:
        scan_embedding = scan_embeddings[index, :]
        sim = 0.5 + 0.5 * np.dot(scan_embedding[np.newaxis, :], report_embeddings.T)[0]
        sort_index = np.argsort(sim)[::-1]
        
        identity_labels = image_identity_labels[index]

        if len(np.intersect1d(np.array(identity_labels), sort_index[:1])):
            img2txt['topK(1)'] += 1
        
        if len(np.intersect1d(np.array(identity_labels), sort_index[:5])):
            img2txt['topK(5)'] += 1
        
        if len(np.intersect1d(np.array(identity_labels), sort_index[:10])):
            img2txt['topK(10)'] += 1

    txt2img = {'topK(1)': torch.zeros([]).to(device), 
               'topK(5)': torch.zeros([]).to(device), 
               'topK(10)': torch.zeros([]).to(device)}
    
    for index in indexes:
        report_embedding = report_embeddings[index, :]
    
        if index not in report_identity_indexes:
            continue

        sim = 0.5 + 0.5 * np.dot(report_embedding[np.newaxis, :], scan_embeddings.T)[0]
        sort_index = np.argsort(sim)[::-1]

        identity_labels = image_identity_labels[index]

        if len(np.intersect1d(np.array(identity_labels), sort_index[:1])):
            txt2img['topK(1)'] += 1
        
        if len(np.intersect1d(np.array(identity_labels), sort_index[:5])):
            txt2img['topK(5)'] += 1
        
        if len(np.intersect1d(np.array(identity_labels), sort_index[:10])):
            txt2img['topK(10)'] += 1
    
    torch.distributed.barrier()
    torch.distributed.all_reduce(img2txt['topK(1)'], op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(img2txt['topK(5)'], op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(img2txt['topK(10)'], op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(txt2img['topK(1)'], op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(txt2img['topK(5)'], op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(txt2img['topK(10)'], op=torch.distributed.ReduceOp.SUM)
    torch.distributed.barrier()

    return {'i2t':{
                    'topK(1)': f"{(img2txt['topK(1)'].item() / scan_embeddings.shape[0] * 100):.3f}%",
                    'topK(5)': f"{(img2txt['topK(5)'].item() / scan_embeddings.shape[0] * 100):.3f}%",
                    'topK(10)': f"{(img2txt['topK(10)'].item() / scan_embeddings.shape[0] * 100):.3f}%",
                }, 
            't2i':{
                    'topK(1)': f"{(txt2img['topK(1)'].item() / len(report_identity_indexes) * 100):.3f}%",
                    'topK(5)': f"{(txt2img['topK(5)'].item() / len(report_identity_indexes) * 100):.3f}%",
                    'topK(10)': f"{(txt2img['topK(10)'].item() / len(report_identity_indexes) * 100):.3f}%",
                }, 
            }

def test_one_epoch(model, data_loader, device, epoch, 
                   image_identity_labels, report_identity_indexes,
                   log_writer=None, args=None):
    model.eval()

    scan_report_features = {}

    if args is not None and hasattr(args, 'retrieval_keys'):
        for key in args.retrieval_keys:
            scan_report_features[key] = {}
            scan_report_features[key]['scan'] = torch.zeros((len(data_loader.dataset), 512), dtype=torch.float32).to(device)
            scan_report_features[key]['report'] = torch.zeros((len(data_loader.dataset), 512), dtype=torch.float32).to(device)
    else:
        scan_report_features['global'] = {}
        scan_report_features['global']['scan'] = torch.zeros((len(data_loader.dataset), 512), dtype=torch.float32).to(device)
        scan_report_features['global']['report'] = torch.zeros((len(data_loader.dataset), 512), dtype=torch.float32).to(device)

    for batch in data_loader:
        with torch.no_grad():            
            feature_outputs = model(batch, mask_ratio=0.0, is_training=False)
            for key in feature_outputs.keys():
                if key in scan_report_features.keys():
                    scan_report_features[key]['scan'][batch['index']] = feature_outputs[key]['scan']
                    scan_report_features[key]['report'][batch['index']] = feature_outputs[key]['report']
    
    for key in scan_report_features.keys():
        torch.distributed.barrier()
        torch.distributed.all_reduce(scan_report_features[key]['scan'], op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(scan_report_features[key]['report'], op=torch.distributed.ReduceOp.SUM)
        torch.distributed.barrier()

    results_output = {}
    for key in scan_report_features.keys():
        scan_features = torch.nn.functional.normalize(scan_report_features[key]['scan'], p=2, dim=-1)
        report_features = torch.nn.functional.normalize(scan_report_features[key]['report'], p=2, dim=-1)
    
        results = retrieval(scan_features.cpu().numpy(), report_features.cpu().numpy(), 
                            image_identity_labels, report_identity_indexes, 
                            indexes=[index for index in range(int(device.split(':')[-1]), 
                                                              len(data_loader.dataset), 
                                                               misc.get_world_size())], 
                            device=device)
    
        if log_writer is not None:
            log_writer.add_scalars(f'retrieval/s2r_{key}', {'topK@1': float(results['i2t']['topK(1)'].strip('%')),
                                                            'topK@5': float(results['i2t']['topK(5)'].strip('%')),
                                                            'topK@10': float(results['i2t']['topK(10)'].strip('%'))},
                                    epoch)

            log_writer.add_scalars(f'retrieval/r2s_{key}', {'topK@1': float(results['t2i']['topK(1)'].strip('%')),
                                                            'topK@5': float(results['t2i']['topK(5)'].strip('%')),
                                                            'topK@10': float(results['t2i']['topK(10)'].strip('%'))},
                                    epoch)
        
        results_output[key] = results
    

    return results_output