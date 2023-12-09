import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict

import discrimintor
from models.epsnet import *
from utils.datasets import *
from utils.datasets import PackedConformationDataset
from utils.transforms import *
from utils.misc import *
from discrimintor import *
from discrimintor import model, runner, utils


def num_confs(num: str):
    if num.endswith('x'):
        return lambda x: x * int(num[:-1])
    elif int(num) > 0:
        return lambda x: int(num)
    else:
        raise ValueError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint',
                        default="log/geodiff/checkpoints/qm9_default.pt")
    parser.add_argument('--disc_config_path', type=str, help='path of dataset', default="configs/qm9_dg_default.yml")

    parser.add_argument('--save_traj', action='store_true', default=False,
                        help='whether store the whole trajectory for sampling')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('2x'))
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=800)
    parser.add_argument('--end_idx', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=5000,
                        help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=0.5,
                        help='enable global gradients only when noise is low')
    parser.add_argument('--w_global', type=float, default=0.3,
                        help='weight for global gradients')
    parser.add_argument('--w_dg', type=float, default=0.5,
                        help='weight for discriminator scores')
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='ld',
                        help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    parser.add_argument('--seed', type=int, default=2021,
                        help='')
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt)
    config_path = "configs/qm9_default.yml"
    # config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), '*.yml'))[0]
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    seed_all(args.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    # load discriminator model and checkpoints
    with open(args.disc_config_path, 'r') as f:
        disc_config = yaml.safe_load(f)
    disc_config = EasyDict(disc_config)

    # Logging
    # output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)
    output_dir = get_new_log_dir(log_dir, str(args.seed)+'wdg'+str(args.w_dg), tag=args.tag)
    # output_dir = "log/"
    logger = get_logger('test', output_dir)
    logger.info(args)

    discriminator_model = discrimintor.model.SDE(disc_config)
    gpus = list(filter(lambda x: x is not None, disc_config.train.gpus))
    solver = runner.DefaultRunner(None, None, None, None, discriminator_model, None,
                                  None, gpus, disc_config)

    dg_model = solver.load_dg(disc_config.test.init_checkpoint, disc_epoch=disc_config.test.epoch)
    # Datasets and loaders
    logger.info('Loading datasets...')
    transforms = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config.model.edge_order),  # Offline edge augmentation
    ])
    if args.test_set is None:
        test_set = PackedConformationDataset(config.dataset.test, transform=transforms)
    else:
        test_set = PackedConformationDataset(args.test_set, transform=transforms)

    # Model
    logger.info('Loading model...')
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])



    test_set_selected = []
    for i, data in enumerate(test_set):
        if not (args.start_idx <= i < args.end_idx): continue
        test_set_selected.append(data)

    done_smiles = set()
    results = []
    if args.resume is not None:
        with open(args.resume, 'rb') as f:
            results = pickle.load(f)
        for data in results:
            done_smiles.add(data.smiles)

    for i, data in enumerate(tqdm(test_set_selected)):
        if data.smiles in done_smiles:
            logger.info('Molecule#%d is already done.' % i)
            continue

        num_refs = data.pos_ref.size(0) // data.num_nodes
        num_samples = args.num_confs(num_refs)

        data_input = data.clone()
        data_input['pos_ref'] = None
        batch = repeat_data(data_input, num_samples).to(args.device)
        dg_model = dg_model.to(args.device)
        clip_local = None
        for _ in range(2):  # Maximum number of retry
            try:
                pos_init = torch.randn(batch.num_nodes, 3).to(args.device)
                pos_gen, pos_gen_traj = model.langevin_dynamics_sample_dg(
                    dg_model,
                    batch,
                    atom_type=batch.atom_type,
                    pos_init=pos_init,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=False,  # Done in transforms.
                    n_steps=args.n_steps,
                    step_lr=1e-6,
                    w_global=args.w_global,
                    w_dg=args.w_dg,
                    global_start_sigma=args.global_start_sigma,
                    clip=args.clip,
                    clip_local=clip_local,
                    sampling_type=args.sampling_type,
                    eta=args.eta
                )
                pos_gen = pos_gen.cpu()
                if args.save_traj:
                    data.pos_gen = torch.stack(pos_gen_traj)
                else:
                    data.pos_gen = pos_gen
                results.append(data)
                done_smiles.add(data.smiles)

                save_path = os.path.join(output_dir, 'samples_%d.pkl' % i)
                logger.info('Saving samples to: %s' % save_path)
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)

                break  # No errors occured, break the retry loop
            except FloatingPointError:
                clip_local = 20
                logger.warning('Retrying with local clipping.')

    save_path = os.path.join(output_dir, 'samples_all.pkl')
    logger.info('Saving samples to: %s' % save_path)
    # logger.info('w_dg = ' % args.w_dg)


    def get_mol_key(data):
        for i, d in enumerate(test_set_selected):
            if d.smiles == data.smiles:
                return i
        return -1


    results.sort(key=get_mol_key)

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
