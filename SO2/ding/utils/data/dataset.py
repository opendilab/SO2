from math import gamma
from random import randint
from typing import List, Dict
import pickle
import torch
import numpy as np
from tqdm import tqdm
import logging
from easydict import EasyDict
from torch.utils.data import Dataset

from ding.utils import DATASET_REGISTRY, import_module


@DATASET_REGISTRY.register('naive')
class NaiveRLDataset(Dataset):

    def __init__(self, cfg) -> None:
        assert type(cfg) in [str, EasyDict], "invalid cfg type: {}".format(type(cfg))
        if isinstance(cfg, EasyDict):
            self._data_path = cfg.policy.collect.data_path
        elif isinstance(cfg, str):
            self._data_path = cfg
        with open(self._data_path, 'rb') as f:
            self._data: List[Dict[str, torch.Tensor]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]


@DATASET_REGISTRY.register('d4rl')
class D4RLDataset(Dataset):

    def __init__(self, cfg: dict) -> None:
        import gym
        import logging
        from time import sleep,time
        import random
        import d4rl
        # for i in range(100):
        #     try:
        #         import d4rl  # register d4rl enviroments with open ai gym
        #     except Exception as e:
        #         random.seed(time())
        #         sleep(random.randint(1,60))
        #         print(e)
        # try:
        #     import d4rl  # register d4rl enviroments with open ai gym
        # except ImportError:
        #     logging.warning("not found d4rl env, please install it, refer to https://github.com/rail-berkeley/d4rl")

        # Init parameters
        data_path = cfg.policy.collect.get('data_path', None)
        env_id = cfg.env.env_id
        # Create the environment
        if data_path:
            d4rl.set_dataset_path(data_path)
        env = gym.make(env_id)
        if cfg.env.get('with_q_value',None):
            dataset,self._q_min,self._q_max = d4rl.qlearning_dataset_with_q(env, gamma=cfg.policy.learn.discount_factor, norm=cfg.policy.learn.get('normalize_q',None))
        else:
            dataset = d4rl.qlearning_dataset(env)
        self._data = []
        self._dataset = dataset
        if cfg.policy.learn.get('normalize_states', None):
            self._normalize_states(dataset)
        if cfg.policy.learn.get('local_q_target', None):
            dataset = self._state_action_local_target_q(dataset,num_clusters=cfg.policy.learn.local_q_target_num_clusters,
            quantile=cfg.policy.learn.local_q_target_quantile)
        self._next_action = cfg.policy.collect.get('next_action', None)
        self._load_d4rl(cfg, dataset)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]

    def _load_d4rl(self,cfg: dict, dataset: Dict[str, np.ndarray]) -> None:
        for i in range(len(dataset['observations'])):
            trans_data = {}
            trans_data['obs'] = torch.from_numpy(dataset['observations'][i])
            trans_data['next_obs'] = torch.from_numpy(dataset['next_observations'][i])
            trans_data['action'] = torch.from_numpy(dataset['actions'][i])
            trans_data['reward'] = torch.tensor(dataset['rewards'][i])
            trans_data['done'] = dataset['terminals'][i]
            if cfg.env.get('with_q_value',None):
                trans_data['q_value'] = torch.tensor(dataset['q_values'][i])
            if cfg.policy.learn.get('local_q_target', None):
                trans_data['q_target'] = torch.tensor(dataset['q_target'][i])
            if self._next_action:
                trans_data['next_action'] = torch.from_numpy(dataset['actions'][(i+1)%len(dataset['observations'])])
            trans_data['collect_iter'] = 0
            self._data.append(trans_data)

    def _normalize_states(self, dataset, eps=1e-5):
        self._mean = dataset['observations'].mean(0, keepdims=True)
        self._std = dataset['observations'].std(0, keepdims=True) + eps
        dataset['observations'] = (dataset['observations'] - self._mean) / self._std
        dataset['next_observations'] = (dataset['next_observations'] - self._mean) / self._std
        self._mean = torch.from_numpy(self._mean)
        self._std = torch.from_numpy(self._std)

    def _state_action_local_target_q(self, dataset, num_clusters=1000, quantile=0.6):
        # init
        state = dataset['observations']
        action = dataset['actions']
        q_value = dataset['q_values']
        data = np.concatenate([state, action], axis=1)
        # cluster
        import sklearn
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=10240, random_state=0).fit(data)
        q_target_list = [q_value[kmeans.labels_==i] for i in range(num_clusters)]
        # sort
        [q.sort() for q in q_target_list]
        q_target = [q[int(quantile*len(q))] for q in q_target_list]
        # assign
        dataset['q_target'] = np.array(q_target)[kmeans.labels_]
        return dataset

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def data(self):
        return self._data
    
    @property
    def raw_dataset(self):
        return self._dataset

@DATASET_REGISTRY.register('d4rl_space')
class D4RLSpaceDataset(Dataset):

    def __init__(self, cfg: dict) -> None:
        import gym
        import random
        import d4rl

        # Init parameters
        data_path = cfg.policy.collect.get('data_path', None)
        env_id = cfg.env.env_id
        self.k = cfg.env.k

        # Create the environment
        if data_path:
            d4rl.set_dataset_path(data_path)
        env = gym.make(env_id)
        dataset = env.get_dataset()
        timeouts = dataset['timeouts'][:-1]
        dataset = d4rl.qlearning_dataset(env, dataset=dataset, terminate_on_end=True)
        dataset['timeouts'] = timeouts

        self._load_d4rl(cfg, dataset)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]

    def _load_d4rl(self,cfg: dict, dataset: Dict[str, np.ndarray]) -> None:
        obs = torch.from_numpy(dataset['observations'])
        next_obs = torch.from_numpy(dataset['next_observations'])
        action = torch.from_numpy(dataset['actions'])
        reward = torch.from_numpy(dataset['rewards'])
        done = torch.from_numpy(dataset['terminals']) & ~torch.from_numpy(dataset['timeouts'])
        self._mean = obs.mean(0, keepdim=True)
        self._std = obs.std(0, keepdim=True)
        self._obs_shape = obs.shape[-1]
        self._action_shape = action.shape[-1]
        action_space, dist_space = self._generate_action_space(obs, action, obs)
        next_action_space, next_dist_space = self._generate_action_space(next_obs, action, obs)
        self._data = [{'obs': obs[i], 'next_obs': next_obs[i], 'action': action[i], 'reward': reward[i], 'done': done[i], 'action_space': action_space[i], 'next_action_space': next_action_space[i], 'dist_space': dist_space[i], 'next_dist_space': next_dist_space[i]} for i in range(len(obs))]

    def _generate_action_space(self, obs, action, obs_set):
        import faiss
        obs = ((obs - self._mean) / self._std).numpy()
        obs_set = ((obs_set - self._mean) / self._std).numpy()
        action_space = []
        dist_space = []

        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(self._obs_shape)
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(obs_set)
        batch_size = 1024
        for i in tqdm(range(0, len(obs), batch_size)):
            dist, k_id = index.search(obs[i: i + batch_size], self.k)
            action_space.append(action[torch.from_numpy(k_id).long()])
            dist_space.append(torch.from_numpy(dist / np.sqrt(self._obs_shape)))

        return torch.cat(action_space, dim=0), torch.cat(dist_space, dim=0)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def data(self):
        return self._data

@DATASET_REGISTRY.register('hdf5')
class HDF5Dataset(Dataset):

    def __init__(self, cfg: dict) -> None:
        try:
            import h5py
        except ImportError:
            logging.warning("not found h5py package, please install it trough 'pip install h5py' ")
        data_path = cfg.policy.collect.get('data_path', None)
        data = h5py.File(data_path, 'r')
        self._load_data(data)
        if cfg.policy.collect.get('normalize_states', None):
            self._normalize_states()

    def __len__(self) -> int:
        return len(self._data['obs'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: self._data[k][idx] for k in self._data.keys()}

    def _load_data(self, dataset: Dict[str, np.ndarray]) -> None:
        self._data = {}
        for k in dataset.keys():
            logging.info(f'Load {k} data.')
            self._data[k] = dataset[k][:]

    def _normalize_states(self, eps=1e-3):
        self._mean = self._data['obs'].mean(0, keepdims=True)
        self._std = self._data['obs'].std(0, keepdims=True) + eps
        self._data['obs'] = (self._data['obs'] - self._mean) / self._std
        self._data['next_obs'] = (self._data['next_obs'] - self._mean) / self._std

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std


def hdf5_save(exp_data, expert_data_path):
    try:
        import h5py
    except ImportError:
        logging.warning("not found h5py package, please install it trough 'pip install h5py' ")
    import numpy as np
    dataset = dataset = h5py.File('%s_demos.hdf5' % expert_data_path.replace('.pkl', ''), 'w')
    dataset.create_dataset('obs', data=np.array([d['obs'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('action', data=np.array([d['action'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('reward', data=np.array([d['reward'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('done', data=np.array([d['done'] for d in exp_data]), compression='gzip')
    dataset.create_dataset('collect_iter', data=np.array([d['collect_iter'] for d in exp_data]), compression='gzip')
    dataset.create_dataset('next_obs', data=np.array([d['next_obs'].numpy() for d in exp_data]), compression='gzip')


def naive_save(exp_data, expert_data_path):
    with open(expert_data_path, 'wb') as f:
        pickle.dump(exp_data, f)


def offline_data_save_type(exp_data, expert_data_path, data_type='naive'):
    globals()[data_type + '_save'](exp_data, expert_data_path)


def create_dataset(cfg, **kwargs) -> Dataset:
    cfg = EasyDict(cfg)
    import_module(cfg.get('import_names', []))
    return DATASET_REGISTRY.build(cfg.policy.collect.data_type, cfg=cfg, **kwargs)
