# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from collections import OrderedDict

from mmfewshot.detection.datasets import (build_dataloader, build_dataset,
                                          get_copy_dataset_type)
from mmfewshot.detection.models import build_detector
import ipdb 
from splits import get_asd_zero_shot_class_ids

def copy_synthesised_weights(model, filename, dataset_name='coco', split='65_8_7'):
    print(filename)
    checkpoint = torch.load(filename, map_location='cpu')
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))
    
    if hasattr(model, 'module'):
        own_state = model.module.state_dict()
    else:
        own_state = model.state_dict()
    use_zero_shot_bg_param = True
    # ipdb.set_trace()
    if use_zero_shot_bg_param:
        filename = 'checkpoints/asd_65_8_7/2022-05-11-zero-shot/classifier_best.pth'
        cp = torch.load(filename, map_location='cpu')
        state_dict['fc1.weight'][-1] = cp['state_dict']['fc1.weight'][-1]
        state_dict['fc1.bias'][-1] = cp['state_dict']['fc1.bias'][-1]
    zero_shot_class_inds = get_asd_zero_shot_class_ids(dataset=dataset_name, split=split)
    
    seen_bg_weights = own_state['roi_head.bbox_head.fc_cls.weight'][-1].data.cpu().numpy().copy()
    seen_bg_bias = own_state['roi_head.bbox_head.fc_cls.bias'][-1].data.cpu().numpy().copy()

    own_state['roi_head.bbox_head.fc_cls.bias'][zero_shot_class_inds] = state_dict['fc1.bias'][:len(zero_shot_class_inds)]
    own_state['roi_head.bbox_head.fc_cls.weight'][zero_shot_class_inds] = state_dict['fc1.weight'][:len(zero_shot_class_inds)]
    # ipdb.set_trace()
    
    alpha1 = 0.35
    alpha2 = 0.65
    own_state['roi_head.bbox_head.fc_cls.bias'][-1] = alpha1*own_state['roi_head.bbox_head.fc_cls.bias'][-1] + alpha2*state_dict['fc1.bias'][-1]
    own_state['roi_head.bbox_head.fc_cls.weight'][-1] = alpha1*own_state['roi_head.bbox_head.fc_cls.weight'][-1] + alpha2*state_dict['fc1.weight'][-1]

    return seen_bg_weights, seen_bg_bias

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFewShot test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--syn_weights', help='the dir to save logs and models')
    parser.add_argument('--asd', action='store_true', help='test only for unseen classes')    
    parser.add_argument('--gasd', action='store_true', help='test for base classes with unseen classes')
    parser.add_argument('--dataset', default='coco', help='coco, objects365')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
        args.cfg_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.show \
        or args.show_dir, (
            'Please specify at least one operation (save/eval/show the '
            'results / save the results) with the argument "--out", "--eval"',
            '"--show" or "--show-dir"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.model.test_cfg.rcnn.asd = args.asd
    cfg.model.test_cfg.rcnn.gasd = args.gasd
    cfg.model.test_cfg.rcnn.score_thr = 0.05

    # currently only support single images testing
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    assert samples_per_gpu == 1, 'currently only support single images testing'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    # pop frozen_parameters
    cfg.model.pop('frozen_parameters', None)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    dataset_name = args.dataset
    if args.syn_weights:
        seen_bg_weight, seen_bg_bias = copy_synthesised_weights(model, args.syn_weights, dataset_name, split=cfg.data.test.split)
        model.roi_head.bbox_head.seen_bg_weight = torch.from_numpy(seen_bg_weight).cuda()
        model.roi_head.bbox_head.seen_bg_bias = torch.from_numpy(seen_bg_bias).cuda()
    # for meta-learning methods which require support template dataset
    # for model initialization.
    if cfg.data.get('model_init', None) is not None:
        cfg.data.model_init.pop('copy_from_train_dataset')
        model_init_samples_per_gpu = cfg.data.model_init.pop(
            'samples_per_gpu', 1)
        model_init_workers_per_gpu = cfg.data.model_init.pop(
            'workers_per_gpu', 1)
        if cfg.data.model_init.get('ann_cfg', None) is None:
            assert checkpoint['meta'].get('model_init_ann_cfg',
                                          None) is not None
            cfg.data.model_init.type = \
                get_copy_dataset_type(cfg.data.model_init.type)
            cfg.data.model_init.ann_cfg = \
                checkpoint['meta']['model_init_ann_cfg']
        model_init_dataset = build_dataset(cfg.data.model_init)
        # disable dist to make all rank get same data
        model_init_dataloader = build_dataloader(
            model_init_dataset,
            samples_per_gpu=model_init_samples_per_gpu,
            workers_per_gpu=model_init_workers_per_gpu,
            dist=False,
            shuffle=False)
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        show_kwargs = dict(show_score_thr=args.show_score_thr)
        
        if cfg.data.get('model_init', None) is not None:
            from mmfewshot.detection.apis import (single_gpu_model_init,
                                                  single_gpu_test)
            single_gpu_model_init(model, model_init_dataloader)
        else:
            # from mmdet.apis.test import single_gpu_test
            from mmfewshot.detection.apis import single_gpu_test
        # args.show = True
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        if cfg.data.get('model_init', None) is not None:
            from mmfewshot.detection.apis import (multi_gpu_model_init,
                                                  multi_gpu_test)
            multi_gpu_model_init(model, model_init_dataloader)
        else:
            from mmdet.apis.test import multi_gpu_test
        outputs = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
        )

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
