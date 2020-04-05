# Continual Learning methods using Episodic Memory

This project provides simple PyTorch-based APIs for continual machine learning methods that use episodic memory. Currently, this supports following continual learning algorithms:

* GEM ([original code](https://github.com/facebookresearch/GradientEpisodicMemory), [paper](https://arxiv.org/abs/1706.08840))
* A-GEM ([original code](https://github.com/facebookresearch/agem), [paper](https://arxiv.org/abs/1812.00420))
* ER (Experience Replay) ([original code](https://github.com/facebookresearch/agem), [paper](https://arxiv.org/abs/1902.10486))

## Prerequisites

* Python 3.6
* PyTorch
* quadprog

## Usage

* All the supported continual learning methods are encapsulated in a class, each supporting the following APIs:
	* `<learner>.prepare()` - sets the optimizer; need to be called prior to training on a task
	* `<learner>.run()` - optimize on a single batch; where the continual learning algorithm is actually run
	* `<learner>.remember()` - add more data to a FIFO memory buffer; input data must be a PyTorch Dataset

* Sample:
	```
	from learners import GEM, AGEM, ER

	memory_capacity = 10240
	task_memory_size = 2048
	memory_sample_size = 64

	# instantiate learner
	learner = AGEM(model, criterion, device=device,
		       memory_capacity=memory_capacity, memory_sample_sz=memory_sample_size)

	# assign optimizer to learner
	learner.prepare(optimizer=torch.optim.Adam, lr=learning_rate)

	model.train()
	for ep in tqdm(range(num_epochs)):
	    for inputs, labels in train_loader:
		# optimize on a single batch
		learner.run(inputs, labels)

	# save data
	learner.remember(train_data, min_save_sz=task_memory_size)
	```

## Experimentation

To test the APIs and to see how the implemented continual learning methods help solve the *catastrophic forgetting* problem, we test each method against a dataset susceptible to such problem. In particular, we use the MNIST dataset, split the training set into 5 sets of equal size, with each having a different class distribution (we'll discuss this further later). We treat each split of the training set as a single learning task.

The target for each learning method is to progressively get higher accuracy on MNIST dataset as it trains successively on each of the 5 tasks. We use the accuracy on the final task as a measure of the method's capability to learn. For comparability, we use a common test set across all methods on which we report the accuracy values. We also measure the algorithm's performance in terms of execution duration.

Apart from the accuracy of the continual learning algorithms, we also measure the accuracy of "offline"/non-continual training to serve as the "gold standard" for learning. We also measure the final accuracy in a continual learning setting where no special algorithms are used; hence, we call it as "Naive Continual" learning.

All throughout the experiment, a neural network with a single hidden layer is used, with hand-picked hyperparameter settings. The whole experiment can be run in [test.ipynb](test.ipynb).

*Note that this was not meant to be an exhaustive evalution of continual learning methods. Thus, the results shall be taken with a grain of salt. :)*

**Offline/Non-continual Baseline**: 95.80%

For a continual learning setup, we simulate two scenarios:

### Case 1: Skewed Splits

In this test, we split the data such that each split or task is comprised dominantly of 2 classes, and only few of the other 8 classes. In particular, each task shall consist 90% of all the training samples of 2 classes, while getting only 2.5% of the remaining classes. This simulates the scenario where there is a defined set of classes, but the influx of data is uneven among the classes, resulting to unbalanced datasets for each learning task.

|Method|Accuracy|Duration (s)|
|---|---|---|
|Naive Continual|84.63%|8.89|
|GEM|95.42%|42.27|
|A-GEM|89.26%|15.64|
|ER|93.88%|14.51|

### Case 2: Class Splits

In contrast to the previous test, in this we use 100% of 2 classes for each task. This also means that each task shall consist only of 2 classes. This simulates an *incremental class learning* problem, where new classes are added in new tasks.

|Method|Accuracy|Duration (s)|
|---|---|---|
|Naive Continual|19.38%|9.46|
|GEM|93.85%|42.50|
|A-GEM|55.36%|15.58|
|ER|86.96%|13.99|


## To do

* fill in more comments

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
