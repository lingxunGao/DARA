"""Original end-to-end training kernel, isolated from historical search launchers."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from . import protocol
from .data import Windows, MixedSampler
from .model import ModelConfig, StructuredDisasterModel
from .horizon_curriculum import horizon_weights, weighted_mask_mean, balanced_prediction_loss

def mean_mask(values, mask):
    return values[mask].mean() if bool(mask.any()) else values.sum()*0


def prediction_loss(prediction, target, group, scale):
    error = (prediction.squeeze(-1)-target.squeeze(-1)).abs()/scale
    valid = target.squeeze(-1) != 0
    losses = [error[valid & (group == g)].mean() for g in range(3) if bool((valid & (group == g)).any())]
    balanced = torch.stack(losses).mean() if losses else error.sum()*0
    return .5*mean_mask(error, valid)+.5*balanced


def state_hash(state, prefix=''):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name.startswith(prefix):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CachedSplit:
    """Materialize exactly the existing Windows inputs once, then index on GPU."""
    def __init__(self, data, split, device):
        self.dataset = Windows(data, split)
        batches = list(DataLoader(self.dataset, batch_size=256))
        self.values = {}
        for key in batches[0]:
            if key == 'inputs':
                self.values[key] = {k: torch.cat([b[key][k] for b in batches]).to(device)
                                    for k in batches[0][key]}
            else:
                self.values[key] = torch.cat([b[key] for b in batches]).to(device)
        self.size = len(self.dataset)
        self.device = device

    def get(self, indices):
        index = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return {key: {k: v[index] for k, v in value.items()} if isinstance(value, dict)
                else value[index] for key, value in self.values.items()}

    def sequential(self, batch_size):
        for start in range(0, self.size, batch_size):
            yield self.get(range(start, min(start+batch_size, self.size)))


def training_loss(model, batch, config, epoch=1, *, return_details=False):
    prediction, details = model(batch['inputs'], unclipped_backbone=True)
    target = batch['target']
    valid = target != 0
    curriculum = config.get('horizon_curriculum')
    weights = (horizon_weights(epoch, target.shape[1], curriculum['end_epoch'], curriculum['floor'])
               if curriculum else None)
    # Return exactly to the established objective once all horizons are equal.
    if weights is not None and all(w == 1. for w in weights):
        weights = None
    if config.get('detach_reference', False):
        # The structured response/context/gate have no parameters shared with
        # the backbone. Only the reference objective may update the backbone.
        if model.config.response != 'structured':
            raise ValueError('Reference gradient isolation requires structured response')
        prediction = (details['base'].detach()*(1+details['gate']*details['response']))[..., None]
    error = (prediction-target).abs()/model.backbone.std
    global_loss = (weighted_mask_mean(error, valid, weights) if weights is not None
                   else error[valid].mean() if bool(valid.any()) else error.sum()*0)
    balanced_weight = config['balanced_weight']
    loss = (1-balanced_weight)*global_loss
    if balanced_weight:
        balanced = (balanced_prediction_loss(prediction, target, batch['group'], model.node_scale, weights)
                    if weights is not None else prediction_loss(prediction, target, batch['group'], model.node_scale))
        loss = loss + balanced_weight*balanced
    if config.get('mse_weight', 0.):
        squared = ((prediction-target)/model.backbone.std).square()
        squared_loss = (weighted_mask_mean(squared, valid, weights) if weights is not None
                        else squared[valid].mean() if bool(valid.any()) else squared.sum()*0)
        loss = loss + config['mse_weight']*squared_loss
    normal = batch['normal'] & valid[..., 0]
    reference_weight = config.get('reference_weight', 0.)
    if reference_weight:
        reference_error = (details['base']-target[..., 0]).abs()/model.node_scale
        reference_loss = reference_error[normal].mean() if bool(normal.any()) else reference_error.sum()*0
        loss = loss + reference_weight*reference_loss
    denoising_weight = config.get('denoising_weight', 0.)
    if denoising_weight:
        # Synthetic input corruption on normal-proxy histories, supervised by
        # their ORIGINAL observed normal futures. No invented disaster labels
        # or counterfactual targets; the response branch never sees this view.
        augmented = dict(batch['inputs'])
        flow = augmented['flow']
        eligible = normal.any(1)
        chosen = (torch.rand_like(flow[:, 0]) < .25) & eligible
        starts = torch.randint(0, 5, chosen.shape, device=flow.device)
        hours = torch.arange(6, device=flow.device)[None, :, None]
        mask = chosen[:, None] & (hours >= starts[:, None]) & (hours < starts[:, None]+2)
        factor = .3+.5*torch.rand_like(flow[:, :1])
        augmented['flow'] = torch.where(mask, flow*factor, flow)
        reference = model.backbone.forward_unclipped(augmented, model.graph_bias)
        error = (reference-target[..., 0]).abs()/model.node_scale
        loss = loss + denoising_weight*(error[normal].mean() if bool(normal.any()) else error.sum()*0)
    regularization = config['regularization']
    if regularization:
        if bool(normal.any()) and not config.get('normal_aux_free', False):
            loss = loss + regularization*(details['gate'][normal].square().mean()
                    + (details['gate']*details['response'])[normal].square().mean())
        if 'context' in details and model.config.response == 'structured':
            loss = loss + regularization*model.shape_penalty(details['context'])
    return (loss, prediction, details) if return_details else loss


def selection_value(score, config):
    return score['mae']+config.get('selection_rmse_weight', 0.)*score['rmse']


def epoch_batches(train, config, sampler, epoch):
    """Epoch-indexed RNG makes the sampling stream replayable after resume."""
    if config['mixed_sampling']:
        sampler.epoch = epoch-1
        return list(sampler)
    order = np.random.default_rng(config['seed']+epoch).permutation(train.size)
    return [order[i:i+config['batch_size']] for i in range(0, train.size, config['batch_size'])]


@torch.no_grad()
def validate(model, cached, batch_size):
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=cached.device)
    group_totals = torch.zeros(3, 2, dtype=torch.float64, device=cached.device)
    normalized_total = torch.zeros((), dtype=torch.float64, device=cached.device)
    for batch in cached.sequential(batch_size):
        prediction = model(batch['inputs'])[0]
        valid = batch['target'] != 0
        error_full = prediction.double()-batch['target'].double()
        error = error_full[valid]
        totals += torch.stack((error.abs().sum(), error.square().sum(), valid.sum().double()))
        normalized = error_full[..., 0].abs()/model.node_scale.double()
        normalized_total += normalized[valid[..., 0]].sum()
        for group in range(3):
            mask = valid[..., 0] & (batch['group'] == group)
            group_totals[group, 0] += normalized[mask].sum()
            group_totals[group, 1] += mask.sum()
    absolute, squared, count = totals.cpu().tolist()
    if not count:
        raise ValueError('Validation has no nonzero targets')
    groups = group_totals.cpu().tolist()
    group_mae = [total/n if n else None for total, n in groups]
    nonempty = [value for value in group_mae if value is not None]
    node_mae = float(normalized_total)/count
    return dict(mae=absolute/count, rmse=math.sqrt(squared/count), valid_count=int(count),
                node_normalized_mae=node_mae, group_normalized_mae=group_mae,
                group_valid_count=[int(n) for _, n in groups],
                balanced_objective=.5*node_mae+.5*sum(nonempty)/len(nonempty))


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def fit_trial(dataset, data, train, val, config, args):
    if config.get('normal_aux_free', False) and (config.get('reference_weight', 0.)
            or config.get('detach_reference', False) or config.get('denoising_weight', 0.)
            or config['isolate_history']):
        raise ValueError('Normal-aux-free trials require raw histories and no reference objectives')
    output = args.output/dataset/'trials'/config['name']
    result_path = output/'validation.json'
    if result_path.exists():
        saved_result = json.loads(result_path.read_text())
        if saved_result['hyperparameters'] != config:
            raise ValueError('Completed trial configuration mismatch')
        return saved_result
    budget = {k: config[k] for k in ('epochs', 'min_epochs', 'patience')}
    budget_path = args.output/'budget_overrides.json'
    if budget_path.exists():
        budget.update(json.loads(budget_path.read_text()).get(dataset, {}).get(config['name'], {}))
    if not 1 <= budget['min_epochs'] <= budget['epochs'] or budget['patience'] < 1:
        raise ValueError('Invalid effective training budget')
    output.mkdir(parents=True, exist_ok=True)
    protocol.seed_everything(config['seed'])
    model_config = ModelConfig(nodes=len(data['preprocessing']['node_names']),
        history=data['preprocessing'].get('history', 6), horizon=data['preprocessing'].get('horizon', 6),
        backbone=config.get('backbone', 'stae_gru'), response=config.get('response', 'structured'), memory=True,
        gate=config.get('gate', True), response_curve=config.get('response_curve', 'piecewise'),
        response_library=config.get('response_library', 'manual'),
        context_fusion=config.get('context_fusion', 'original'),
        locality_mode=config.get('locality_mode', 'original'),
        mechanism_mode=config.get('mechanism_mode', 'none'),
        operator_mode=config.get('operator_mode', 'none'),
        operator_static=config.get('operator_static', False),
        spatial_experts=config.get('spatial_experts', 'none'),
        gate_parameter_mode=config.get('gate_parameter_mode', 'fixed'),
        condition_response=config.get('condition_response', True),
        shared_response_parameters=config.get('shared_response_parameters', False),
        mlp_response_bound=config.get('mlp_response_bound', 1.),
        dynamic_mode=config.get('dynamic_mode','flow'),cross_region_topk=config.get('cross_region_topk',0),
        hidden=config['hidden'], layers=config['layers'], dropout=config['dropout'],
        stae_layers=config['stae_layers'], gru_hidden=config['gru_hidden'],
        short_steps=config['short_steps'], isolate_history=config['isolate_history'],
        contextual_response=config.get('contextual_response', False))
    model = StructuredDisasterModel(model_config, data['preprocessing']).to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if not all(p.requires_grad for p in model.backbone.parameters()):
        raise AssertionError('Backbone must be trainable')
    initial_hash = state_hash(model.state_dict(), 'backbone.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    if config['schedule'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=.5, patience=12, threshold=1e-4,
            min_lr=config['lr']*.01)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=budget['epochs'], eta_min=config['lr']*.01)
    sampler = MixedSampler(train.dataset, config['batch_size'], config['seed'])
    manifest = dict(dataset=dataset, config=model.configuration(), hyperparameters=config,
        initialization='random; no pretrained weights; all parameters trained from epoch 1',
        data=data['manifest'], parameters=sum(p.numel() for p in model.parameters()),
        backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
        initial_backbone_sha256=initial_hash, selection='full validation MAE; zero targets masked',
        initial_model_sha256=state_hash(model.state_dict()),
        initial_shared_model_sha256=state_hash({k:v for k,v in model.state_dict().items()
            if not k.startswith(('context.edge_gate.', 'context.null_read.', 'context.spatial_policies.'))
            and not k.startswith('gate.dynamic_')
            and not any(part in k for part in ('parameter_adapter.', 'temporal_state.', 'hourly_relations.'))}),
        horizon_curriculum=config.get('horizon_curriculum'),
        training='all training targets; joint end-to-end; unclipped linear outputs only for training loss',
        backbone_semantics='jointly trained forecasting component, not an identified normal counterfactual',
        mixed_sampler=dict(enabled=config['mixed_sampling'], cutoff=sampler.cutoff,
            high_windows=len(sampler.high), ordinary_windows=len(sampler.ordinary),
            fallback=sampler.fallback, fit='training windows only; peak exposure over input and target hours',
            rule='half high / half ordinary per batch, with replacement; permutation if a pool is empty'),
        prediction_objective=dict(balanced_weight=config['balanced_weight'],
            formula='(1-w)*global_MAE/train_std + w*(0.5*node_normalized_MAE + 0.5*mean_nonempty_group_MAE)',
            groups=['normal_proxy', 'low_exposure', 'high_exposure'],
            group_thresholds='training statistics only; zero targets masked; empty groups skipped'),
        reference_supervision=dict(weight=config.get('reference_weight', 0.),
            detach_from_response_loss=config.get('detach_reference', False),
            denoising_weight=config.get('denoising_weight', 0.),
            target='observed normal-proxy targets with normal local input history',
            corruption='25% eligible nodes; two consecutive input hours scaled by U(0.3,0.8); original targets'),
        torch_version=str(torch.__version__), numpy_version=str(np.__version__), smoke=args.smoke)
    protocol.write_json(output/'manifest.json', manifest)
    if config.get('normal_aux_free', False):
        manifest['reference_supervision'] = dict(weight=0., detach_from_response_loss=False, denoising_weight=0.)
        manifest['prediction_objective']['groups'] = ['low_physical_exposure', 'medium_physical_exposure', 'high_physical_exposure']
        manifest['prediction_objective']['mse_weight'] = config.get('mse_weight', 0.)
    manifest['selection'] = f"full validation MAE + {config.get('selection_rmse_weight', 0.)} * RMSE"
    manifest['effective_budget'] = budget
    protocol.write_json(output/'manifest.json', manifest)
    initial_score = validate(model, val, config['batch_size'])
    best_score, best_epoch, bad_epochs, first_epoch = selection_value(initial_score, config), 0, 0, 1
    history = []

    def save_best(epoch, score):
        atomic_save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                         config=model.configuration(), preprocessing=data['preprocessing'],
                         manifest=manifest, epoch=epoch, validation=score), output/'model.pt')

    latest_path = output/'latest.pt'
    if latest_path.exists():
        saved = torch.load(latest_path, map_location=args.device, weights_only=False)
        if saved['hyperparameters'] != config:
            raise ValueError('Partial trial configuration mismatch')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        best_score, best_epoch, bad_epochs = saved['best_score'], saved['best_epoch'], saved['bad_epochs']
        first_epoch = saved['epoch']+1
        history = saved['history']
        torch.set_rng_state(saved['rng'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    else:
        save_best(0, initial_score)
    start_run = time.monotonic()
    for epoch in range(first_epoch, budget['epochs']+1):
        begin = time.monotonic()
        model.train()
        batches = epoch_batches(train, config, sampler, epoch)
        losses = []
        backbone_gradient = None
        sampled_windows, high_windows = 0, 0
        group_counts = torch.zeros(3, dtype=torch.long, device=args.device)
        for indices in batches:
            batch = train.get(indices)
            sampled_windows += len(indices)
            high_windows += int(np.isin(indices, sampler.high).sum())
            for group in range(3):
                group_counts[group] += ((batch['target'][..., 0] != 0) & (batch['group'] == group)).sum()
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, batch, config, epoch=epoch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Non-finite loss: {dataset}/{config["name"]}/{epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            if backbone_gradient is None:
                readout = (model.backbone.network.output.weight if model.config.backbone in ('dynamic_gcru', 'dynamic_original')
                           else model.backbone.transformer.output_proj.weight)
                backbone_gradient = float(readout.grad.norm())
            optimizer.step()
            losses.append(loss.detach())
            if args.smoke and len(losses) == 2:
                break
        score = validate(model, val, config['batch_size'])
        if config['schedule'] == 'plateau':
            scheduler.step(selection_value(score, config))
        else:
            scheduler.step()
        if selection_value(score, config) < best_score:
            best_score, best_epoch, bad_epochs = selection_value(score, config), epoch, 0
            save_best(epoch, score)
        else:
            bad_epochs += 1
        row = dict(epoch=epoch, train_loss=float(torch.stack(losses).mean()),
                   horizon_loss_weights=(horizon_weights(epoch, model_config.horizon, config['horizon_curriculum']['end_epoch'],
                       config['horizon_curriculum']['floor']) if config.get('horizon_curriculum') else [1.]*model_config.horizon),
                   validation_mae=score['mae'], validation_rmse=score['rmse'],
                   validation_balanced_objective=score['balanced_objective'],
                   validation_group_normalized_mae=score['group_normalized_mae'],
                   sampled_windows=sampled_windows, sampled_high_windows=high_windows,
                   sampled_ordinary_windows=sampled_windows-high_windows,
                   train_group_valid_count=group_counts.cpu().tolist(),
                   best_selection_score=best_score, best_epoch=best_epoch,
                   learning_rate=optimizer.param_groups[0]['lr'],
                   backbone_readout_gradient_norm=backbone_gradient, seconds=time.monotonic()-begin)
        history.append(row)
        with (output/'training.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if epoch % 5 == 0 or epoch == 1:
            print(json.dumps(dict(dataset=dataset, trial=config['name'], **row)), flush=True)
        atomic_save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), epoch=epoch, history=history,
            best_score=best_score, best_epoch=best_epoch, bad_epochs=bad_epochs,
            hyperparameters=config, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), latest_path)
        if epoch >= budget['min_epochs'] and bad_epochs >= budget['patience']:
            break
    selected = torch.load(output/'model.pt', map_location='cpu', weights_only=False)
    selected['manifest']['effective_budget'] = budget
    atomic_save(selected, output/'model.pt')
    selected_hash = state_hash(selected['state_dict'], 'backbone.')
    if selected['epoch'] > 0 and selected_hash == initial_hash:
        raise AssertionError('Selected backbone never changed')
    result = dict(dataset=dataset, trial=config['name'], best_selection_score=best_score,
                  best_validation_mae=selected['validation']['mae'],
                  best_validation_rmse=selected['validation']['rmse'], selected_epoch=best_epoch,
                  best_validation_balanced_objective=selected['validation']['balanced_objective'],
                  best_validation_group_normalized_mae=selected['validation']['group_normalized_mae'],
                  epochs_run=history[-1]['epoch'], hyperparameters=config,
                  effective_budget=budget,
                  model_path=str(output/'model.pt'), backbone_changed=selected_hash != initial_hash,
                  seconds_this_invocation=time.monotonic()-start_run, smoke=args.smoke)
    protocol.write_json(result_path, result)
    protocol.write_json(output/'training.json', history)
    print('TRIAL COMPLETE '+json.dumps(result), flush=True)
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

