# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from .sampled_dataset import SampledDataset
from typing import Dict, List

from utils import utils
from models.BaseModel import BaseModel
from . import BaseReader
import scipy.sparse as sp

class BaseRunner(object):
	@staticmethod
	def parse_runner_args(parser):
		parser.add_argument('--epoch', type=int, default=200,
							help='Number of epochs.')
		parser.add_argument('--check_epoch', type=int, default=1,
							help='Check some tensors every check_epoch.')
		parser.add_argument('--test_epoch', type=int, default=-1,
							help='Print test results every test_epoch (-1 means no print).')
		parser.add_argument('--early_stop', type=int, default=10,
							help='The number of epochs when dev results drop continuously.')
		parser.add_argument('--lr', type=float, default=1e-3,
							help='Learning rate.')
		parser.add_argument('--l2', type=float, default=0,
							help='Weight decay in optimizer.')
		#parser.add_argument('--batch_size', type=int, default=256,
		parser.add_argument('--batch_size', type=int, default=2,
							help='Batch size during training.')
		#parser.add_argument('--eval_batch_size', type=int, default=256,
		parser.add_argument('--eval_batch_size', type=int, default=2,
							help='Batch size during testing.')
		parser.add_argument('--optimizer', type=str, default='Adam',
							help='optimizer: SGD, Adam, Adagrad, Adadelta')
		parser.add_argument('--num_workers', type=int, default=5,
							help='Number of processors when prepare batches in DataLoader')
		parser.add_argument('--pin_memory', type=int, default=0,
							help='pin_memory in DataLoader')
		#parser.add_argument('--topk', type=str, default='5,10,20,50',
		parser.add_argument('--topk', type=str, default='1',
							help='The number of items recommended to each user.')
		parser.add_argument('--metric', type=str, default='NDCG,HR',
							help='metrics: NDCG, HR')
		parser.add_argument('--main_metric', type=str, default='',
							help='Main metric to determine the best model.')
		return parser

	@staticmethod
	def evaluate_method(predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
		"""
		:param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
		:param topk: top-K value list
		:param metrics: metric string list
		:return: a result dict, the keys are metric@topk
		"""
		evaluations = dict()
		# sort_idx = (-predictions).argsort(axis=1)
		# gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
		# ↓ As we only have one positive sample, comparing with the first item will be more efficient. 
		gt_rank = (predictions >= predictions[:,0].reshape(-1,1)).sum(axis=-1)
		# if (gt_rank!=1).mean()<=0.05: # maybe all predictions are the same
		# 	predictions_rnd = predictions.copy()
		# 	predictions_rnd[:,1:] += np.random.rand(predictions_rnd.shape[0], predictions_rnd.shape[1]-1)*1e-6
		# 	gt_rank = (predictions_rnd > predictions[:,0].reshape(-1,1)).sum(axis=-1)+1
		
	
		for k in topk:
			hit = (gt_rank <= k)
			for metric in metrics:
				key = '{}@{}'.format(metric, k)
				if metric == 'HR':
					evaluations[key] = hit.mean()
				elif metric == 'NDCG':
					evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
				else:
					raise ValueError('Undefined evaluation metric: {}.'.format(metric))
		return evaluations


	def _build_adj_matrix(self):
		"""Construct the adjacency matrix using the training data."""
		interactions = self.data_df['train'][['user_id', 'item_id']]
		rows = interactions['user_id'].values
		cols = interactions['item_id'].values + self.n_users
		data = np.ones_like(rows)
		self.adj_matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.n_users + self.n_items, self.n_users + self.n_items))
		logging.info("Adjacency matrix built with shape {}".format(self.adj_matrix.shape))

	def get_encoder_adj(self):
		"""Return the normalized adjacency matrix for the encoder."""
		rowsum = np.array(self.adj_matrix.sum(1)).flatten()  # Compute row sums
		rowsum[rowsum == 0] = 1  # Replace zeros with ones to prevent division by zero
		d_inv = np.power(rowsum, -0.5)  # Compute inverse square root
		d_mat_inv = sp.diags(d_inv)  # Create diagonal matrix
		norm_adj_tmp = d_mat_inv.dot(self.adj_matrix)
		encoder_adj = norm_adj_tmp.dot(d_mat_inv)
		return encoder_adj


	def get_sub_adj(self):
		"""Return a sub adjacency matrix by sampling edges."""
		rows, cols = self.adj_matrix.nonzero()
		edge_count = len(rows)
		keep_count = int(edge_count * 0.1)  # Keep 10% of edges
		keep_indices = np.random.choice(edge_count, size=keep_count, replace=False)
		sub_rows = rows[keep_indices]
		sub_cols = cols[keep_indices]
		sub_data = np.ones_like(sub_rows)
		sub_adj_matrix = sp.csr_matrix((sub_data, (sub_rows, sub_cols)), shape=self.adj_matrix.shape)
		return sub_adj_matrix

	def get_cmp_adj(self):
		"""Return a comparison adjacency matrix with additional edges."""
		rows, cols = self.adj_matrix.nonzero()
		additional_edges = int(len(rows) * 0.1)  # Add 10% more edges
		add_rows = np.random.choice(self.n_users, size=additional_edges)
		add_cols = np.random.choice(self.n_items, size=additional_edges) + self.n_users
		cmp_rows = np.concatenate([rows, add_rows])
		cmp_cols = np.concatenate([cols, add_cols])
		cmp_data = np.ones_like(cmp_rows)
		cmp_adj_matrix = sp.csr_matrix((cmp_data, (cmp_rows, cmp_cols)), shape=self.adj_matrix.shape)
		return cmp_adj_matrix

	def __init__(self, args):
		self.train_models = args.train
		self.epoch = args.epoch
		self.check_epoch = args.check_epoch
		self.test_epoch = args.test_epoch
		self.early_stop = args.early_stop
		self.learning_rate = args.lr
		self.batch_size = args.batch_size
		self.eval_batch_size = args.eval_batch_size
		self.l2 = args.l2
		self.optimizer_name = args.optimizer
		self.num_workers = args.num_workers
		self.pin_memory = args.pin_memory
		self.topk = [int(x) for x in args.topk.split(',')]
		self.metrics = [m.strip().upper() for m in args.metric.split(',')]
		self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0]) if not len(args.main_metric) else args.main_metric # early stop based on main_metric
		self.main_topk = int(self.main_metric.split("@")[1])
		self.time = None  # will store [start_time, last_step_time]

		self.log_path = os.path.dirname(args.log_file) # path to save predictions
		self.save_appendix = args.log_file.split("/")[-1].split(".")[0] # appendix for prediction saving
		self._build_adj_matrix()

	def _check_time(self, start=False):
		if self.time is None or start:
			self.time = [time()] * 2
			return self.time[0]
		tmp_time = self.time[1]
		self.time[1] = time()
		return self.time[1] - tmp_time

	def _build_optimizer(self, model):
		logging.info('Optimizer: ' + self.optimizer_name)
		optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
			model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
		return optimizer

	def train(self, data_dict: Dict[str, BaseModel.Dataset]):
		model = data_dict['train'].model
		main_metric_results, dev_results = list(), list()
		self._check_time(start=True)
		try:
			for epoch in range(self.epoch):
				# Fit
				self._check_time()
				gc.collect()
				torch.cuda.empty_cache()
				loss = self.fit(data_dict['train'], epoch=epoch + 1)
				if np.isnan(loss):
					logging.info("Loss is Nan. Stop training at %d."%(epoch+1))
					break
				training_time = self._check_time()

				# Observe selected tensors
				if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
					utils.check(model.check_list)

				# Record dev results
				dev_result = self.evaluate(data_dict['dev'], [self.main_topk], self.metrics)
				dev_results.append(dev_result)
				main_metric_results.append(dev_result[self.main_metric])
				logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	dev=({})'.format(
					epoch + 1, loss, training_time, utils.format_metric(dev_result))

				# Test
				if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
					test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
					logging_str += ' test=({})'.format(utils.format_metric(test_result))
				testing_time = self._check_time()
				logging_str += ' [{:<.1f} s]'.format(testing_time)

				# Save model and early stop
				if max(main_metric_results) == main_metric_results[-1] or \
						(hasattr(model, 'stage') and model.stage == 1):
					model.save_model()
					logging_str += ' *'
				logging.info(logging_str)

				if self.early_stop > 0 and self.eval_termination(main_metric_results):
					logging.info("Early stop at %d based on dev result." % (epoch + 1))
					break

		except KeyboardInterrupt:
			logging.info("Early stop manually")
			exit_here = input("Exit completely without evaluation? (y/n) (default n):")
			if exit_here.lower().startswith('y'):
				logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
				exit(1)

		# Find the best dev result across iterations
		best_epoch = main_metric_results.index(max(main_metric_results))
		logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
			best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
		model.load_model()

	def sample_nodes(self, num_nodes, sample_size):
		"""Randomly sample a subset of nodes."""
		sampled_nodes = np.random.choice(num_nodes, size=sample_size, replace=False)
		return sampled_nodes

	def create_node_subgraph(self, adj, sampled_nodes):
		"""Create a subgraph with sampled nodes."""
		rows, cols = adj.nonzero()
		mask = np.isin(rows, sampled_nodes) & np.isin(cols, sampled_nodes)
		sub_rows = rows[mask]
		sub_cols = cols[mask]
		sub_data = adj.data[mask]
		sub_adj = sp.csr_matrix((sub_data, (sub_rows, sub_cols)), shape=adj.shape)
		return sub_adj
	
	def sample_neighborhood(self, adj, sampled_nodes, depth=2):
		"""Expand a sampled set of nodes to include their neighbors up to a certain depth."""
		subgraph = adj[sampled_nodes]
		for _ in range(depth - 1):
			neighbors = subgraph.nonzero()[1]
			subgraph = adj[neighbors]
		return subgraph


	def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
		model = dataset.model
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		dataset.actions_before_epoch()  # Must sample before multi-thread start

		model.train()
		loss_lst = list()

		for batch_idx in range(len(dataset) // self.batch_size):  # Loop through batches
			# Apply sampling strategies for each batch
			num_nodes = self.n_users + self.n_items
			sample_size = int(num_nodes * 0.1)  # Use 10% of nodes
			sampled_nodes = self.sample_nodes(num_nodes, sample_size)

			# Create subgraphs based on sampled nodes
			sampled_adj = self.create_node_subgraph(self.adj_matrix, sampled_nodes)
			sampled_cmp = self.create_node_subgraph(self.adj_matrix, sampled_nodes)
			sampled_encoder_adj = self.create_node_subgraph(self.adj_matrix, sampled_nodes)

			# Convert to torch sparse tensors
			sub_adj = self.csr_to_torch_sparse(sampled_adj)
			cmp_adj = self.csr_to_torch_sparse(sampled_cmp)
			encoder_adj = self.csr_to_torch_sparse(sampled_encoder_adj)

			sampled_dataset = SampledDataset(
				original_dataset=dataset,
				adj_matrix=self.adj_matrix,
				n_users=self.n_users,
				n_items=self.n_items,
				sample_ratio=0.001  # Adjust the sampling ratio as needed
			)
			
			# DataLoader for the current sampled batch
			dl = DataLoader(sampled_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
							collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)

			for batch in tqdm(dl, leave=False, desc='Epoch {:<3}, Batch {:<3}'.format(epoch, batch_idx), ncols=100, mininterval=1):
				batch = utils.batch_to_gpu(batch, model.device)

				# Randomly shuffle items to avoid overfitting
				item_ids = batch['item_id']
				indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)						
				batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]

				model.optimizer.zero_grad()

				# Forward pass with sampled data
				out_dict = model(
					handler=None,  # Modify this if handler is needed
					is_test=False,
					sub=sub_adj,
					cmp=cmp_adj,
					encoderAdj=encoder_adj
				)

				# Shuffle predictions back to match the original order
				prediction = out_dict['prediction']
				if len(prediction.shape) == 2:  # Only for ranking tasks
					restored_prediction = torch.zeros(*prediction.shape).to(prediction.device)
					restored_prediction[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices] = prediction   
					out_dict['prediction'] = restored_prediction

				# Compute loss and backpropagate
				loss = model.loss(out_dict)
				loss.backward()
				model.optimizer.step()
				loss_lst.append(loss.detach().cpu().data.numpy())

		return np.mean(loss_lst).item()


	def eval_termination(self, criterion: List[float]) -> bool:
		if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
			return True
		elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
			return True
		return False

	def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric@k)
		"""
		sampled_dataset  = SampledDataset(
				original_dataset=dataset,
				adj_matrix=self.adj_matrix,
				n_users=self.n_users,
				n_items=self.n_items,
				sample_ratio=0.001  # Adjust the sampling ratio as needed
			)
		
		predictions = self.predict(sampled_dataset)
		return self.evaluate_method(predictions, topks, metrics)

	def csr_to_torch_sparse(self, csr_mat):
		"""Convert a scipy.sparse.csr_matrix to a torch sparse tensor."""
		coo = csr_mat.tocoo()
		indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
		values = torch.tensor(coo.data, dtype=torch.float32)
		shape = coo.shape
		return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()
	
	#def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> np.ndarray:
	#def predict(self, dataset: BaseReader, save_prediction: bool = False) -> np.ndarray:
	def predict(self, dataset: BaseReader, save_prediction: bool = False) -> np.array:
		"""
		The returned prediction is a 2D-array, each row corresponds to all the candidates,
		and the ground-truth item poses the first.
		Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
				 predictions like: [[1,3,4], [2,5,6]]
		"""
		dataset.model.eval()
		predictions = list()
		dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
			"""
			if hasattr(dataset.model,'inference'):
				prediction = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			else:
				prediction = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			"""
			###
			#handler = dataset.handler  # Retrieve handler from dataset
			#sub = handler.get_sub_adj()  # Ensure these methods return the required matrices
			#cmp = handler.get_cmp_adj()
			#encoderAdj = handler.get_encoder_adj()
			#sub = dataset.get_sub_adj()
			#cmp = dataset.get_cmp_adj()
			#encoderAdj = dataset.get_encoder_adj()
			sub = self.csr_to_torch_sparse(dataset.get_sub_adj())
			cmp = self.csr_to_torch_sparse(dataset.get_cmp_adj())
			encoderAdj = self.csr_to_torch_sparse(dataset.get_encoder_adj())

			# Pass all required arguments to the forward function
			prediction = dataset.model(
				handler=None,
				is_test=True,
				sub=sub,
				cmp=cmp,
				encoderAdj=encoderAdj,
			)['prediction']

			###
			predictions.extend(prediction.cpu().data.numpy())
		predictions = np.array(predictions)

		if dataset.model.test_all:
			rows, cols = list(), list()
			for i, u in enumerate(dataset.data['user_id']):
				clicked_items = list(dataset.corpus.train_clicked_set[u] | dataset.corpus.residual_clicked_set[u])
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf
		return predictions

	def print_res(self, dataset: BaseModel.Dataset) -> str:
		"""
		Construct the final result string before/after training
		:return: test result string
		"""
		result_dict = self.evaluate(dataset, self.topk, self.metrics)
		res_str = '(' + utils.format_metric(result_dict) + ')'
		return res_str
