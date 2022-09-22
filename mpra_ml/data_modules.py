import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


BASE_TO_IDX = {
	"A": 0,
	"C": 1,
	"G": 2,
	"T": 3,
	"pad": 4 # Padding
} 
IDX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "pad"}
STR_INDICATOR_TO_IDX = {
	"non-STR": 0,
	"STR": 1,
	"pad": 2,
}


class STRDataset(Dataset):
	"""Dataset for STR data.

	Args:
		str_dataframe: DataFrame with STR data. Must have columns:
			'left_flank': Sequence of bases to the left of the STR motif
			'motif': Sequence of bases that make up the STR motif
			'right_flank': Sequence of bases to the right of the STR motif
			'nrep_ref': Number of repeats of the motif in the reference
				sequence
			label_col: Column in str_dataframe to use as label, matching 
				label_col arg
		label_col (str): Column in str_dataframe to use as label.
		str_variation (str or None): Type of variation to use for STRs. 
			Options:
				None (default): Use reference sequence
				'uniform': Use uniform distribution of number of repeats
					ranging from reference number +/- variation_range, where
					variation_range is specified in str_variation_kwargs. If
					min_motif_repeats is specified in str_variation_kwargs,
					then the number of repeats will be at least that number.
		str_variation_kwargs (dict): Keyword arguments for str_variation.
			Options:
				'min_motif_repeats' (int): Minimum number of repeats of motif.
					Defaults to 1 if not specified.
				for 'uniform': 
					variation_range (int): Range of number of repeats
		return_data (bool): If True, __getitem__ include the row from the
			DataFrame in output dict with key 'data'.
	"""

	def __init__(
		self,
		str_dataframe,
		label_col="active",
		str_variation=None,
		str_variation_kwargs={},
		return_data=False,
	):
		self.label_col = label_col
		self.str_variation = str_variation
		self.str_variation_kwargs = str_variation_kwargs
		self.return_data = return_data

		self.str_data = str_dataframe

		# Create dicts that map from indices in the dataframe to that STR's
		# sequences as integer indices. This is the format the model uses as
		# input.
		self.left_flank_seqs = self.str_data.left_flank.map(
			lambda x: torch.tensor([BASE_TO_IDX[base] for base in x])
		).to_dict()
		self.right_flank_seqs = self.str_data.right_flank.map(
			lambda x: torch.tensor([BASE_TO_IDX[base] for base in x])
		).to_dict()
		self.motif_seqs = self.str_data.motif.map(
			lambda x: torch.tensor([BASE_TO_IDX[base] for base in x])
		).to_dict()

		# Setup opitions for str_variation
		if self.str_variation is not None:
			if self.str_variation == 'uniform':
				self.str_variation_range = self.str_variation_kwargs['variation_range']
				if 'min_motif_repeats' in self.str_variation_kwargs:
					self.min_motif_repeats = self.str_variation_kwargs['min_motif_repeats']
				else:
					self.min_motif_repeats = 1

	def __len__(self):
		return self.str_data.shape[0]

	def _get_n_repeats(self, nrep):
		"""Returns number of repeats to use for STR.

		Args:
			nrep (int): Number of repeats in reference sequence

		Returns:
			int: Number of repeats to use for STR
		"""
		if self.str_variation is None:
			return nrep
		elif self.str_variation == 'uniform':
			return np.random.randint(
				max(nrep - self.str_variation_range, self.min_motif_repeats),
				nrep + self.str_variation_range + 1
			)
		else:
			raise NotImplementedError(
				"str_variation {} not implemented".format(self.str_variation)
			)

	def make_sequence(self, idx):
		"""Returns integer representation of STR sequence.

		Args:
			idx: Index of row in self.str_data to get sequence for

		Returns:
			tuple of (seq, str_indicator) where seq is a torch tensor of
			integers and str_indicator is a torch tensor of 0s and 1s that
			indicates which indices are part of the STR motif.
		"""
		left_flank_seq = self.left_flank_seqs[idx]
		right_flank_seq = self.right_flank_seqs[idx]
		motif_seq = self.motif_seqs[idx]

		# Get number of repeats of motif to use in returned sequence
		motif_seq = motif_seq.repeat(
			self._get_n_repeats(self.str_data.nrep_ref[idx])
		)

		# Create sequence
		seq = torch.cat(
			[left_flank_seq, motif_seq, right_flank_seq],
			dim=0
		)

		# Create indicator vector
		str_indicator = torch.cat(
			[
				torch.ones(left_flank_seq.shape[0]) * STR_INDICATOR_TO_IDX["non-STR"],
				torch.ones(motif_seq.shape[0]) * STR_INDICATOR_TO_IDX["STR"],
				torch.ones(right_flank_seq.shape[0]) * STR_INDICATOR_TO_IDX["non-STR"],
			],
			dim=0
		)

		return seq, str_indicator

	def __getitem__(self, idx):
		"""Returns single sample with index idx.

		Returns:
			dict with keys:
				'data': Row from self.str_data DataFrame
				'label': From label_col
				'seq': Interger representation of STR sequence returned by
					make_sequence()
				'str_indicator': Integer vector the same length as seq that
					indicates which indices are part of the STR motif. Also
					returned by make_sequence()
		"""
		seq, str_indicator = self.make_sequence(idx)
		str_row = self.str_data.iloc[idx]

		ret_dict = {
			"label": str_row[self.label_col],
			"seq": seq,
			"str_indicator": str_indicator,
		}

		if self.return_data:
			ret_dict["data"] = str_row

		return ret_dict


def STR_data_collate_fn(batch):
	"""Collate function for STRDataset.

	Args:
		batch: List of dicts from STRDateset.__getitem__ with keys 'data',
			'label', 'seq', and 'str_indicator'

	Returns:
		dict with keys 'data', 'label', and 'input'. 'input' is a dict with
		keys 'seq' and 'str_indicator' that are padded and stacked to form
		tensors.
	"""
	includes_data = "data" in batch[0].keys()
	if includes_data:
		batch_data = [item["data"] for item in batch]
	batch_label = torch.tensor([item["label"] for item in batch]).long()
	batch_seqs = [item["seq"] for item in batch]
	batch_str_indicators = [item["str_indicator"] for item in batch]

	# Pad sequences
	batch_seqs = torch.nn.utils.rnn.pad_sequence(
		batch_seqs,
		batch_first=True,
		padding_value=BASE_TO_IDX["pad"]
	).long()
	batch_str_indicators = torch.nn.utils.rnn.pad_sequence(
		batch_str_indicators,
		batch_first=True,
		padding_value=STR_INDICATOR_TO_IDX["pad"]
	).long()

	ret_dict = {
		"label": batch_label,
		"input": {
			"seq": batch_seqs,
			"str_indicator": batch_str_indicators,
		}
	}

	if includes_data:
		ret_dict["data"] = batch_data

	return ret_dict


class STRDataModule(pl.LightningDataModule):
	""" Data Module for loading data splits for training/eval.

	Args:
		split_col (str): Column name that has 0/1/2 train/val/test split labels.

	"""
	def __init__(
		self,
		str_data_path,
		split_col,
		label_col="active",
		str_variation=None,
		str_variation_kwargs={},
		return_data=False,
		batch_size=32,
		num_workers=0,
	):
		super().__init__()
		self.str_data_path = str_data_path
		self.split_col = split_col
		self.label_col = label_col
		self.str_variation = str_variation
		self.str_variation_kwargs = str_variation_kwargs
		self.return_data = return_data
		self.batch_size = batch_size
		self.num_workers = num_workers

	def setup(self, stage=None):
		str_df = pd.read_csv(self.str_data_path)

		self.train_data = STRDataset(
			str_df[str_df[self.split_col] == 0].reset_index(),
			label_col=self.label_col,
			str_variation=self.str_variation,
			str_variation_kwargs=self.str_variation_kwargs,
			return_data=self.return_data
		)
		self.val_data = STRDataset(
			str_df[str_df[self.split_col] == 1].reset_index(),
			label_col=self.label_col,
			str_variation=self.str_variation,
			str_variation_kwargs=self.str_variation_kwargs,
			return_data=self.return_data
		)
		self.test_data = STRDataset(
			str_df[str_df[self.split_col] == 2].reset_index(),
			label_col=self.label_col,
			str_variation=self.str_variation,
			str_variation_kwargs=self.str_variation_kwargs,
			return_data=self.return_data
		)

	def train_dataloader(self):
		return DataLoader(
			self.train_data,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			collate_fn=STR_data_collate_fn,
			shuffle=True,
		)
	
	def val_dataloader(self):
		return DataLoader(
			self.val_data,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			collate_fn=STR_data_collate_fn,
		)

	def test_dataloader(self):
		return DataLoader(
			self.test_data,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			collate_fn=STR_data_collate_fn,
		)


if __name__ == "__main__":
	data_path = os.path.join(
		"..", "..", "data", "preprocessed_data", "preprocessed_data.csv"
	)

	data_df = pd.read_csv(data_path)

	# Create a dataset object
	str_dataset = STRDataset(data_df)

	demo_batch = [str_dataset[i] for i in range(8)]
	collated_batch = STR_data_collate_fn(demo_batch)

	str_ds_w_data = STRDataset(data_df, return_data=True)
	demo_batch = [str_ds_w_data[i] for i in range(8)]
	collated_batch_w_data = STR_data_collate_fn(demo_batch)

	# Create a data module object
	str_data_module = STRDataModule(
		data_path,
		split_col="split_2",
	)
	str_data_module.setup()
	val_dl = str_data_module.val_dataloader()

	# Test variation
	var_str_dataset = STRDataset(
		data_df,
		str_variation="uniform",
		str_variation_kwargs={
			"min_motif_repeats": 2,
			"variation_range": 5
		}
	)
	
	example_str = var_str_dataset.str_data.iloc[0]
	motif_len = len(example_str["motif"])
	ref_repeat_count = example_str["nrep_ref"]
	instances_of_str_0 = [var_str_dataset[0] for _ in range(10)]
	collated_str_0 = STR_data_collate_fn(instances_of_str_0)
	n_str_bases = (collated_str_0['input']['str_indicator'] == 1).sum(axis=1)
	repeat_counts = n_str_bases / motif_len
	print(f"Reference repeat count: {ref_repeat_count}")
	print(repeat_counts)
