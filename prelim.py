import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


	
if __name__ == '__main__':
	design_data_path = 'data/design/array_probes_split.tsv'
	expression_data_path = 'data/expression/characterization.csv'

	# Read in the design data
	design_data = pd.read_csv(design_data_path, sep='\t')

	# Read in the expression data
	active_df = pd.read_csv(expression_data_path)
	active_df['id'] = active_df.STR.str.split('_').map(
		lambda x: '_'.join(x[:-1])
	)
	active_df['active_bin'] = active_df.active.map(lambda x: x == 'active').astype(int)

	print(active_df.groupby('id').active_bin.mean().value_counts())