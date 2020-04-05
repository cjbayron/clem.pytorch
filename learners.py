# learners

from torch.utils.data import ConcatDataset, Subset, DataLoader
import torch.nn as nn
import torch

import gradutils

class Learner(nn.Module):
    ''' Base class for all learners
    '''
    def __init__(self, model, criterion, device):
        '''
        '''
        super(Learner, self).__init__()

        self.model = model
        self.criterion = criterion
        self.device = device

    def prepare(self, optimizer, **args):
        ''' Sets optimizer

        Call this to set optimizer and "refresh" state of learner.
        '''
        self.optimizer = optimizer(self.model.parameters(), **args)

    def run(self, inputs, labels, optimize_weights=True):
        '''
        '''
        # calculate gradients
        self.zero_grad()
        self.optimizer.zero_grad()
        out = self.model(inputs)
        loss = self.criterion(out, labels)
        loss.backward()

        if optimize_weights:
            self.optimizer.step()


class EpisodicMemoryLearner(Learner):
    ''' Base class of episodic memory-based continual learners
    '''
    def __init__(self, model, criterion, memory_capacity, memory_sample_sz, device):
        '''
        '''
        super(EpisodicMemoryLearner, self).__init__(model, criterion, device)

        self.memory = None # will be initialized as PyTorch Dataset
        self.memory_capacity = memory_capacity # total capacity
        self.memory_sample_sz = memory_sample_sz # sampling size to use when fetching from memory
        self.memory_loader = None

    def remember(self, data, min_save_sz, fill_buffer=False):
        ''' Push data to memory buffer

        data: instance of PyTorch Dataset
        min_save_sz: Minimum number of samples to save from `data`
        fill_buffer: If set to True, saves enough samples to fill buffer capacity.
            If available space is less than `min_save_sz`, saves `min_save_sz`
            samples and flushes out buffer data in excess of capacity (in FIFO manner).
            If available space is greater than size of `data`, saves all of `data`
            in buffer.
        '''
        # size checking & correction
        if min_save_sz > self.memory_capacity:
            raise Exception("min_save_sz exceeds memory_capacity!")

        max_save_sz = len(data)
        if max_save_sz < min_save_sz:
            min_save_sz = max_save_sz

        # compute effective save size
        eff_save_sz = min_save_sz
        mem_sz = len(self.memory) if self.memory else 0
        mem_free = self.memory_capacity - mem_sz
        if fill_buffer:
            if mem_free < min_save_sz: # buffer will overflow after adding
                eff_save_sz = min_save_sz
            elif mem_free > max_save_sz: # buffer space can save ALL of `data`
                eff_save_sz = max_save_sz
            else: # fill buffer
                eff_save_sz = mem_free

        # now, handle overflow
        cur_datasets = self.memory.datasets if self.memory else None
        if mem_free < eff_save_sz:
            mem_overflow = eff_save_sz - mem_free # this is the amount of data we will flush
            # find index where we will 'cut' current dataset
            for i, ds in enumerate(cur_datasets):
                if mem_overflow <= len(ds):
                    cur_datasets = cur_datasets[(i+1):]
                    if mem_overflow < len(ds):
                        # number of items to spare in datasets[i]
                        mem_to_spare = len(ds) - mem_overflow
                        indices_to_spare = torch.randperm(len(ds))[:mem_to_spare]
                        spared = Subset(ds, indices=indices_to_spare)
                        cur_datasets = [spared, *cur_datasets]

                    break

                mem_overflow -= len(ds)

        # get randomly sampled Subset from `data`
        if eff_save_sz < max_save_sz:
            indices_to_save = torch.randperm(max_save_sz)[:eff_save_sz]
            data = Subset(data, indices=indices_to_save)

        if self.memory is None: # initialize memory
            self.memory = ConcatDataset([data])
        else: # concatenate memory
            self.memory = ConcatDataset([*cur_datasets, data])

    def forget_all(self):
        ''' Clear memory
        '''
        self.memory = None
        self.memory_loader = None


class GEM(EpisodicMemoryLearner):
    '''
    @inproceedings{GradientEpisodicMemory,
        title={Gradient Episodic Memory for Continual Learning},
        author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
        booktitle={NIPS},
        year={2017},
        url={https://arxiv.org/abs/1706.08840}
    }

    Code based on: https://github.com/facebookresearch/GradientEpisodicMemory
    License: Attribution-NonCommercial 4.0 International,
             https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/LICENSE

    Code based on: https://github.com/GT-RIPL/Continual-Learning-Benchmark
    License: MIT, https://github.com/GT-RIPL/Continual-Learning-Benchmark/blob/master/LICENSE
    '''
    def __init__(self, model, criterion, memory_capacity=1024, memory_sample_sz=128, device='cpu'):
        '''
        '''
        super(GEM, self).__init__(model, criterion, memory_capacity, memory_sample_sz, device)

    def run(self, inputs, labels):
        '''
        '''
        if self.memory:
            # calculate and save gradients on previous task/s (loaders)
            past_gradients = []
            for loader in self.memory_loaders:
                self.zero_grad()
                # based on a very simple test, this
                # samples UNIFORMLY from the whole memory.
                #
                # note also that in effect, this loop
                # gets only 1 gradient vector from EACH past task
                past_i, past_l = next(iter(loader))

                if self.device.type == 'cuda':
                    past_i, past_l = past_i.cuda(), past_l.cuda()

                past_out = self.model(past_i)
                past_loss = self.criterion(past_out, past_l)

                past_loss.backward()
                # save gradient
                past_gradients.append(gradutils.get_gradient(self.model))

        # calculate gradients on new task
        super(GEM, self).run(inputs, labels, optimize_weights=False)

        # check inequality constraint and project as needed
        if self.memory:
            # (num_grads,)
            cur_gradient = gradutils.get_gradient(self.model)
            # (1 -> max_num_task, num_grads)
            past_gradients = torch.stack(past_gradients)
            # (1, num_grads) x (num_grads, 1 -> max_num_task)
            dotp = torch.mm(cur_gradient.unsqueeze(0), past_gradients.T)
            if (dotp < 0).sum() != 0:
                # calculate new gradients
                new_grad = gradutils.project_gradient_qp(cur_gradient, past_gradients.T)
                gradutils.update_gradient(self.model, new_grad)

        # update weights (using the final gradients)
        self.optimizer.step()

    def remember(self, data, min_save_sz, fill_buffer=False):
        '''
        '''
        super(GEM, self).remember(data, min_save_sz, fill_buffer)
        # treat each dataset as a separate "task"
        self.memory_loaders = []
        for dataset in self.memory.datasets:
            self.memory_loaders.append(DataLoader(dataset, shuffle=True, batch_size=self.memory_sample_sz))


class AGEM(EpisodicMemoryLearner):
    '''
    @inproceedings{AGEM,
      title={Efficient Lifelong Learning with A-GEM},
      author={Chaudhry, Arslan and Ranzato, Marc’Aurelio and Rohrbach, Marcus and Elhoseiny, Mohamed},
      booktitle={ICLR},
      year={2019}
    }

    Code based on: https://github.com/facebookresearch/agem
    License: MIT, https://github.com/facebookresearch/agem/blob/master/LICENSE
    '''
    def __init__(self, model, criterion, memory_capacity=1024, memory_sample_sz=128, device='cpu'):
        '''
        '''
        super(AGEM, self).__init__(model, criterion, memory_capacity, memory_sample_sz, device)

    def run(self, inputs, labels):
        '''
        '''
        if self.memory:
            # based on a very simple test, this
            # samples UNIFORMLY from the whole memory.
            past_i, past_l = next(iter(self.memory_loader))
            if self.device.type == 'cuda':
                past_i, past_l = past_i.cuda(), past_l.cuda()

            past_out = self.model(past_i)
            past_loss = self.criterion(past_out, past_l)

            past_loss.backward()
            # save gradient
            past_gradient = gradutils.get_gradient(self.model)

        # calculate gradients on new task
        super(AGEM, self).run(inputs, labels, optimize_weights=False)

        # check inequality constraint and project as needed
        if self.memory:
            # (num_grads,)
            cur_gradient = gradutils.get_gradient(self.model)
            dotp = torch.dot(cur_gradient, past_gradient) # scalar
            if dotp < 0:
                # efficient gradient projection
                ref_mag = torch.dot(past_gradient, past_gradient)
                new_grad = cur_gradient - ((dotp / ref_mag) * past_gradient)
                gradutils.update_gradient(self.model, new_grad)

        # update weights (using the final gradients)
        self.optimizer.step()

    def remember(self, data, min_save_sz, fill_buffer=False):
        '''
        '''
        super(AGEM, self).remember(data, min_save_sz, fill_buffer)
        # treat all past datasets as single dataset
        self.memory_loader = DataLoader(self.memory, shuffle=True, batch_size=self.memory_sample_sz)


class ER(EpisodicMemoryLearner):
    ''' Experience Replay
    '''
    def __init__(self, model, criterion, memory_capacity=1024, memory_sample_sz=128, device='cpu'):
        '''
        '''
        super(ER, self).__init__(model, criterion, memory_capacity, memory_sample_sz, device)

    def run(self, inputs, labels):
        '''
        '''
        if self.memory:
            # based on a very simple test, this
            # samples UNIFORMLY from the whole memory.
            past_i, past_l = next(iter(self.memory_loader))
            if self.device.type == 'cuda':
                past_i, past_l = past_i.cuda(), past_l.cuda()

            # concatenate memory with current data
            inputs = torch.cat([inputs, past_i], dim=0)
            labels = torch.cat([labels, past_l], dim=0)

        # calculate gradients on new task
        super(ER, self).run(inputs, labels, optimize_weights=True)

    def remember(self, data, min_save_sz, fill_buffer=False):
        '''
        '''
        super(ER, self).remember(data, min_save_sz, fill_buffer)
        # treat all past datasets as single dataset
        self.memory_loader = DataLoader(self.memory, shuffle=True, batch_size=self.memory_sample_sz)
