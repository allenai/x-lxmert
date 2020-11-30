from pathlib import Path
from time import time
from tqdm import tqdm
import numpy as np
import h5py
import faiss
import argparse
import pickle


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_data(feat_path, grid_size=8, d=2048, topk=-1):
    """
    return:
        data (np.array float32): [len(img_ids) * grid_size ** 2, d]
        img_ids
    """
    with h5py.File(feat_path, 'r') as f:
        img_ids = list(f.keys())
        if topk is not None and topk > 0:
            img_ids = img_ids[:topk]

        data = np.zeros((len(img_ids) * grid_size ** 2, d), dtype=np.float32)

        for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids), ncols=150):
            features = np.zeros((grid_size, grid_size, d), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(features)
            features = np.reshape(features, (grid_size**2, d))

            data[i * grid_size ** 2: (i+1) * grid_size ** 2] += features

    return data, img_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_centroids', type=int, default=10000)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--feat_dim', type=int, default=2048)
    parser.add_argument('--grid_size', type=int, default=8)
    parser.add_argument('--imsize', type=int, default=-1)
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument('--encoder', type=str, default='maskrcnn')
    parser.add_argument('--assign_only', action='store_true')
    parser.add_argument('--src', type=str, default='mscoco_train')
    parser.add_argument('--tgt', type=str, action='store', nargs='+', default='mscoco_train')
    parser.add_argument('--datasets_dir', type=str, default='../datasets/')
    args = parser.parse_args()
    print(args)

    datasets_dir = Path(args.datasets_dir).resolve()

    n_centroids = args.n_centroids
    n_iter = args.n_iter
    d = args.feat_dim
    verbose = args.verbose
    grid_size = args.grid_size
    src = args.src
    # tgt = args.tgt
    imsize = args.imsize

    ######## Load Data ########
    feat_path_dict = {
        'mscoco_train': datasets_dir.joinpath(f'COCO/features/{args.encoder}_train_grid{grid_size}.h5'),
        'mscoco_valid': datasets_dir.joinpath(f'COCO/features/{args.encoder}_valid_grid{grid_size}.h5'),
        'nlvr_train': datasets_dir.joinpath(f'nlvr2/features/{args.encoder}_train_grid{grid_size}.h5'),
        'nlvr_valid': datasets_dir.joinpath(f'nlvr2/features/{args.encoder}_valid_grid{grid_size}.h5'),
        'vg': datasets_dir.joinpath(f'VG/features/{args.encoder}_grid{grid_size}.h5'),
    }

    save_dir = datasets_dir.joinpath('cluster_centroids')
    if not save_dir.exists():
        save_dir.mkdir()

    for tgt in args.tgt:
        tgt_feat_path = feat_path_dict[tgt]
        assert Path(tgt_feat_path).is_file(), tgt_feat_path

    if not args.assign_only:
        src_feat_path = feat_path_dict[src]
        assert Path(src_feat_path).is_file(), src_feat_path

        # Load train data
        x_src, src_img_ids = load_data(src_feat_path, grid_size, d)
        print('Train data shape:', x_src.shape)


        ######## Run Clustering ########
        kmeans = faiss.Kmeans(d, n_centroids, niter=n_iter, verbose=verbose)
        print('Clustering start!')

        start = time()
        kmeans.train(x_src)
        elapsed = time() - start
        print(f'It tool {elapsed:.2f}s ({elapsed/3600:.1f}h)')
        print('Centroids shape:', kmeans.centroids.shape)

        if imsize < 0:
            centroid_path = f'{args.encoder}_{src}_centroids{n_centroids}_iter{n_iter}_d{d}_grid{grid_size}.npy'
        else:
            centroid_path = f'{args.encoder}_{src}_centroids{n_centroids}_iter{n_iter}_d{d}_grid{grid_size}_imsize{imsize}.npy'
        centroid_path = save_dir.joinpath(centroid_path)
        centroids = kmeans.centroids
        np.save(centroid_path, centroids)
        print('Centroids saved at', centroid_path)
    else:
        if imsize < 0:
            centroid_path = f'{args.encoder}_{src}_centroids{n_centroids}_iter{n_iter}_d{d}_grid{grid_size}.npy'
        else:
            centroid_path = f'{args.encoder}_{src}_centroids{n_centroids}_iter{n_iter}_d{d}_grid{grid_size}_imsize{imsize}.npy'
        centroid_path = save_dir.joinpath(centroid_path)
        centroids = np.load(centroid_path)
        print('Centroids loaded from', centroid_path)

    ######## Cluster assignment ########
    index = faiss.IndexFlatL2(d)
    index.add(centroids)

    for tgt in args.tgt:
        print(f'Processing {tgt} data..')
        # Load target data
        if tgt == src:
            if args.assign_only:
                x_src, src_img_ids = load_data(src_feat_path, grid_size, d)
            x_tgt = x_src
            tgt_img_ids = src_img_ids
        else:
            tgt_feat_path = feat_path_dict[tgt]
            x_tgt, tgt_img_ids = load_data(tgt_feat_path, grid_size, d)

        print(f'{tgt} data shape:', x_tgt.shape)

        ######## Cluster assignment ########
        print(f'Assign clusters for {tgt} data...')
        D_tgt, I_tgt = index.search(x_tgt, 1)
        if imsize < 0:
            tgt_cluster_path = f'{args.encoder}_{src}_{tgt}_cluster_index_centroids{n_centroids}_iter{n_iter}_d{d}_grid{grid_size}.npy'
        else:
            tgt_cluster_path = f'{args.encoder}_{src}_{tgt}_cluster_index_centroids{n_centroids}_iter{n_iter}_d{d}_grid{grid_size}_imsize{imsize}.npy'

        tgt_cluster_path = save_dir.joinpath(tgt_cluster_path)
        np.save(tgt_cluster_path, I_tgt)
        print('Cluster saved at', tgt_cluster_path)

        ######## Image ID to Cluster ID ########
        img_id_to_cluster_id = {}
        for i, img_id in enumerate(tgt_img_ids):
            img_id_to_cluster_id[img_id] = I_tgt[i * (grid_size**2):
                                                 (i + 1) * (grid_size**2)].squeeze()

        if imsize < 0:
            cluster_id_dict_path = f'{args.encoder}_{src}_{tgt}_img_id_to_cluster_id_{n_centroids}_iter{n_iter}_d{d}_grid{grid_size}.pkl'
        else:
            cluster_id_dict_path = f'{args.encoder}_{src}_{tgt}_img_id_to_cluster_id_{n_centroids}_iter{n_iter}_d{d}_grid{grid_size}_imsize{imsize}.pkl'
        cluster_id_dict_path = save_dir.joinpath(cluster_id_dict_path)
        with open(cluster_id_dict_path, 'wb') as f:
            pickle.dump(img_id_to_cluster_id, f)
        print('Saved img grid -> cluster id dictionary at', cluster_id_dict_path)

        del(x_tgt)
        del(tgt_img_ids)
        del(D_tgt)
        del(I_tgt)
        del(img_id_to_cluster_id)
