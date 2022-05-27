import os, glob, functools, json
import librosa
import torch
from torch.utils.data import Subset, Dataset, DataLoader, random_split, ConcatDataset, SubsetRandomSampler, BatchSampler
import pytorch_lightning as pl
import numpy as np
from diffsynth.f0 import process_f0
from omegaconf.listconfig import ListConfig

def mix_iterable(dl_a, dl_b):
    for i, j in zip(dl_a, dl_b):
        yield i
        yield j

class ReiteratableWrapper():
    def __init__(self, f, length):
        self._f = f
        self.length = length

    def __iter__(self):
        # make generator
        return self._f()

    def __len__(self):
        return self.length

class WaveParamDataset(Dataset):
    def __init__(self, base_dir, sample_rate=16000, length=4.0, params=True, f0=False, ram=False):
        self.base_dir = base_dir
        self.audio_dir = os.path.join(base_dir, 'audio')
        if f0:
            self.f0_dir = os.path.join(base_dir, 'f0')
            assert os.path.exists(self.f0_dir)
        if params:
            self.param_dir = os.path.join(base_dir, 'param')
            assert os.path.exists(self.param_dir)
        raw_files = sorted(glob.glob(os.path.join(self.audio_dir, '*.wav')))
        self.basenames = [os.path.basename(rf)[:-4] for rf in raw_files]
        print('{0} files in dataset'.format(len(self.basenames)))
        self.length = length
        self.sample_rate = sample_rate
        self.params = params
        self.f0 = f0
        self.ram = ram
        if ram:
            print('Loading files to ram...')
            self.dataset = []
            for bn in self.basenames:
                data = self.read_files(bn)
                self.dataset.append(data)
            print('Finished loading to ram')  
    
    def read_files(self, basename):
        wav_file = os.path.join(self.audio_dir, basename+'.wav')
        audio, _sr = librosa.load(wav_file, sr=self.sample_rate, duration=self.length)
        assert audio.shape[0] == self.length * self.sample_rate
        data = {'audio': audio}
        if self.params:
            param_file = os.path.join(self.param_dir, basename+'.pt')
            data['params'] = torch.load(param_file)
        if self.f0:
            f0_file = os.path.join(self.f0_dir, basename+'.pt')
            f0, periodicity = torch.load(f0_file)
            data['BFRQ'] = process_f0(f0, periodicity).unsqueeze(-1)
        return data

    def __getitem__(self, idx):
        if self.ram:
            data = self.dataset[idx]
        else:
            data = self.read_files(self.basenames[idx])
        return data

    def __len__(self):
        return len(self.basenames)

class FilteredNsynthDataset(Dataset):
    def __init__(self, base_dir, filter_args, sample_rate=16000, length=4.0, f0=False):
        self.base_dir = base_dir
        self.audio_dir = os.path.join(base_dir, 'audio')
        self.raw_files = sorted(glob.glob(os.path.join(self.audio_dir, '*.wav')))
        self.length = length
        self.sample_rate = sample_rate
        # load json file that comes with nsynth dataset
        print('{0} files before filtering'.format(len(self.raw_files)))
        with open(os.path.join(self.base_dir, 'examples.json')) as f:
            self.json_dict = json.load(f)
        # restrict the dataset to some categories
        self.filter_dataset(**filter_args)
        self.nb_files = len(self.filtered_keys)
        self.f0 = f0
        if f0:
            self.f0_dir = os.path.join(base_dir, 'f0')
            assert os.path.exists(self.f0_dir)
            # all the f0 files should already be written
            # with the same name as the audio

    def filter_dataset(self, ng_inst_list=[], ng_source_list=[], ng_quality_list=[], pitch_lower=48, pitch_upper=72):
        self.ng_inst_list = ng_inst_list
        self.ng_source_list = ng_source_list
        self.ng_quality_list = ng_quality_list
        self.pitch_range = range(pitch_lower, pitch_upper)
        def filt(entry):
            if entry['instrument_source_str'] in ng_source_list:
                return False
            elif not entry['pitch'] in self.pitch_range:
                return False
            elif entry['instrument_family_str'] in ng_inst_list:
                return False
            elif any([(q in entry['qualities_str']) for q in ng_quality_list]):
                return False
            else:
                return True
        self.filtered_dict = {key:entry for key, entry in self.json_dict.items() if filt(entry)}
        print('{0} files after filtering'.format(len(self.filtered_dict)))
        self.filtered_keys = list(self.filtered_dict.keys())

    def __getitem__(self, index):
        data = {}
        note = self.filtered_dict[self.filtered_keys[index]]
        file_name = os.path.join(self.audio_dir, note['note_str']+'.wav')
        data['audio'], _sr = librosa.load(file_name, sr=self.sample_rate, duration=self.length)
        # f0
        if self.f0:
            f0_file_name = os.path.join(self.f0_dir, note['note_str']+'.pt')
            f0, periodicity = torch.load(f0_file_name)
            f0_hz = process_f0(f0, periodicity)
            data['BFRQ'] = f0_hz.unsqueeze(-1)
        return data

    def __len__(self):
        return len(self.filtered_dict)

class IdOodDataModule(pl.LightningDataModule):
    def __init__(self, id_dir, ood_dir, train_type, batch_size, sample_rate=16000, length=4.0, num_workers=8, splits=[.8, .1, .1], f0=False, seed=0):
        super().__init__()
        self.id_dir = id_dir
        self.ood_dir = ood_dir
        assert train_type in ['id', 'ood', 'mixed']
        self.train_type = train_type
        self.splits = splits
        self.sr = sample_rate
        self.l = length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.f0 = f0
        self.seed = seed
    
    def create_split(self, dataset):
        dset_l = len(dataset)
        split_sizes = [int(dset_l*self.splits[0]), int(dset_l*self.splits[1])]
        split_sizes.append(dset_l - split_sizes[0] - split_sizes[1])
        dset_train, dset_valid, dset_test = random_split(dataset, lengths=split_sizes, generator=torch.Generator().manual_seed(self.seed))
        return {'train': dset_train, 'valid': dset_valid, 'test': dset_test}

    def setup(self, stage):
        id_dat = WaveParamDataset(self.id_dir, self.sr, self.l, True, self.f0)
        id_datasets = self.create_split(id_dat)
        # ood should be the same size as in-domain
        ood_dat = WaveParamDataset(self.ood_dir, self.sr, self.l, False, self.f0)
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(ood_dat), len(id_dat), replace=False)
        ood_dat = Subset(ood_dat, indices)
        ood_datasets = self.create_split(ood_dat)
        self.id_datasets = id_datasets
        self.ood_datasets = ood_datasets
        assert len(id_datasets['train']) == len(ood_datasets['train'])
        if self.train_type == 'mixed':
            dat_len = len(id_datasets['train'])
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(dat_len, dat_len//2, replace=False)
            self.train_set = ConcatDataset([Subset(id_datasets['train'], indices), Subset(ood_datasets['train'], indices)])

    def train_dataloader(self):
        if self.train_type=='id':
            return DataLoader(self.id_datasets['train'], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)
        elif self.train_type=='ood':
            return DataLoader(self.ood_datasets['train'], batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True)
        elif self.train_type=='mixed':
            id_indices = list(range(len(self.train_set)//2))
            ood_indices = list(range(len(self.train_set)//2, len(self.train_set)))
            id_samp = SubsetRandomSampler(id_indices)
            ood_samp = SubsetRandomSampler(ood_indices)
            id_batch_samp = BatchSampler(id_samp, batch_size=self.batch_size, drop_last=False)
            ood_batch_samp = BatchSampler(ood_samp, batch_size=self.batch_size, drop_last=False)
            generator = functools.partial(mix_iterable, id_batch_samp, ood_batch_samp)
            b_sampler = ReiteratableWrapper(generator, len(id_batch_samp)+len(ood_batch_samp))
            return DataLoader(self.train_set, batch_sampler=b_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return [DataLoader(self.id_datasets["valid"], batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False),
                DataLoader(self.ood_datasets["valid"], batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)]

    def test_dataloader(self):
        return [DataLoader(self.id_datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers),
                DataLoader(self.ood_datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers)]

class MultiDataModule(pl.LightningDataModule):
    def __init__(self, data_cfgs, train_key, batch_size, max_dat_size=20000, num_workers=8, splits=[.8, .1, .1], seed=0):
        super().__init__()
        self.datasets = data_cfgs # should be instantiated recursively
        assert (isinstance(train_key, (list, ListConfig)) or train_key in self.datasets), train_key
        self.train_key = train_key
        self.splits = splits
        self.max_dat_size = max_dat_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dats = dict()
        self.seed = seed

    def create_split(self, dataset):
        dset_l = len(dataset)
        split_sizes = [int(dset_l*self.splits[0]), int(dset_l*self.splits[1])]
        split_sizes.append(dset_l - split_sizes[0] - split_sizes[1])
        dset_train, dset_valid, dset_test = random_split(dataset, lengths=split_sizes, generator=torch.Generator().manual_seed(self.seed))
        return {'train': dset_train, 'valid': dset_valid, 'test': dset_test}

    def setup(self, stage):
        for key, dataset in self.datasets.items():
            if len(dataset) > self.max_dat_size:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(len(dataset), self.max_dat_size, replace=False)
                dataset = Subset(dataset, indices)
            self.dats[key] = self.create_split(dataset)
        if isinstance(self.train_key, (list, ListConfig)):
            if len(self.train_key) > 2:
                raise ValueError("MultiDataModule doesn't support training on 3 or more datasets")
            rng = np.random.default_rng(self.seed)
            # halve each dataset
            first_train_len = len(self.dats[self.train_key[0]]['train'])
            first_indices = rng.choice(first_train_len, first_train_len//2, replace=False)
            first_train = Subset(self.dats[self.train_key[0]]['train'], first_indices)
            second_train_len = len(self.dats[self.train_key[1]]['train'])
            second_indices = rng.choice(second_train_len, second_train_len//2, replace=False)
            second_train = Subset(self.dats[self.train_key[1]]['train'], second_indices)
            self.train_set = ConcatDataset([first_train, second_train])

    def train_dataloader(self):
        if isinstance(self.train_key, (list, ListConfig)):
            # train on mix of two datasets
            indices_a = list(range(len(self.train_set)//2)) # first half
            indices_b = list(range(len(self.train_set)//2, len(self.train_set)))
            samp_a = SubsetRandomSampler(indices_a)
            samp_b = SubsetRandomSampler(indices_b)
            batch_samp_a = BatchSampler(samp_a, batch_size=self.batch_size, drop_last=False)
            batch_samp_b = BatchSampler(samp_b, batch_size=self.batch_size, drop_last=False)
            # interleave between batchsamplers each corresponding to a dataset
            generator = functools.partial(mix_iterable, batch_samp_a, batch_samp_b)
            b_sampler = ReiteratableWrapper(generator, len(batch_samp_a)+len(batch_samp_b))
            return DataLoader(self.train_set, batch_sampler=b_sampler, num_workers=self.num_workers, pin_memory=True)
        else:
            return DataLoader(self.dats[self.train_key]['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return [DataLoader(dat['valid'], batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True) for dat in self.dats.values()]

    def test_dataloader(self):
        return [DataLoader(dat['test'], batch_size=self.batch_size, num_workers=self.num_workers) for dat in self.dats.values()]