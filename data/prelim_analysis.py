import os

import numpy as np
import pandas as pd


if __name__ == '__main__':
	data_path = os.path.join('preprocessed_data', 'preprocessed_data.csv')
	data_df = pd.read_csv(data_path)

	# Get lengths of flanks
	data_df['left_flank_len'] = data_df.left_flank.str.len()
	data_df['right_flank_len'] = data_df.right_flank.str.len()

	print(f"Left flank min: {data_df.left_flank_len.min()}\tmax: {data_df.left_flank_len.max()}")
	print(f"Right flank min: {data_df.right_flank_len.min()}\tmax: {data_df.right_flank_len.max()}")

	# Get reference sequence length (left_flank + motif + right_flank)
	data_df['ref_seq_len'] = data_df.left_flank.str.len() + data_df.motif.str.len() * data_df.nrep_ref + data_df.right_flank.str.len()
	# min: 138 max: 168