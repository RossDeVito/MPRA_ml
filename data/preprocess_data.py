import os

import numpy as np
import pandas as pd


if __name__ == '__main__':
	design_data_path = os.path.join('design', 'array_probes_split.tsv')
	expression_data_path = os.path.join('expression', 'characterization.csv')

	# Read in data
	design_df = pd.read_csv(design_data_path, sep='\t')
	expression_df = pd.read_csv(expression_data_path)

	# Reduce design data to only what we need
	design_df = design_df[
		['id', 'motif', 'nrep_ref', 'left_flank', 'right_flank']
	].drop_duplicates()
	# Round nrep_ref to nearest integer
	design_df['nrep_ref'] = design_df.nrep_ref.round().astype(int)

	# Get active labels from characterization.csv
	expression_df['id'] = expression_df.STR.str.split('_').map(
		lambda x: '_'.join(x[:-1])
	)
	expression_df['active'] = expression_df.active.map(lambda x: x == 'active').astype(int)
	active_labels = expression_df.groupby('id').active.max().reset_index()

	# Merge design and expression data
	merged_df = pd.merge(design_df, active_labels, on='id', how='right')
	assert merged_df.shape[0] == active_labels.shape[0]

	# Add random splits to data set for cross-validation
	## split 1: 80% train, 10% val, 10% test
	np.random.seed(147)
	merged_df['split_1'] = np.random.choice(
		[0, 1, 2], 
		merged_df.shape[0], 
		p=[.8, .1, .1]
	)
	## split 2: 75% train, 12.5% val, 12.5% test
	merged_df['split_2'] = np.random.choice(
		[0, 1, 2],
		merged_df.shape[0],
		p=[.75, .125, .125]
	)
	## split 3: 70% train, 15% val, 15% test
	merged_df['split_3'] = np.random.choice(
		[0, 1, 2],
		merged_df.shape[0],
		p=[.7, .15, .15]
	)

	# Save data
	merged_df.to_csv(
		os.path.join('preprocessed_data', 'preprocessed_data.csv'),
		index=False
	)