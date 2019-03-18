import pandas as pd
import numpy as np
import glob

def score_calibration(y_pred):
    thr = 0.932107 #Calculated by validation set for minimize ACER
    a = 0.5/thr
    b = 1
    y_pred = np.array([a*x if x<thr else b*x for x in y_pred])
    return y_pred

def main():
	test_df = pd.read_csv('data/lists/phase2_list.txt')
	exp_groups = [['exp1_2stage_seed1'], ['exp2_seed1', 'exp2_seed2'], ['exp3b_seed1', 'exp3b_seed2'], ['exp3c_seed1', 'exp3c_seed2']]
	tta_columns = ['crop0', 'crop0_hflip']
	test_res = np.empty((0,len(test_df)))
	for exp_seeds in exp_groups:
		seed_res = np.empty((0,len(test_df)))
		for exp_name in exp_seeds:
			exp_res = np.empty((0,len(test_df)))
			files = glob.glob(f'data/opts/{exp_name}/fold*/tta_epoch31_test_score.txt')
			for fname in files:
				curr_df = pd.read_csv(fname)
				curr_res = curr_df[tta_columns].mean(1).values
				exp_res = np.vstack((exp_res, curr_res))
			exp_res = exp_res.mean(0)
			seed_res = np.vstack((seed_res, exp_res))
		
		seed_res = seed_res.mean(0)
		test_res = np.vstack((test_res, seed_res))

	test_res = test_res.mean(0)
	test_res = score_calibration(test_res)
	test_df['score'] = np.around(test_res, decimals=6)
	test_df.to_csv('data/predict_sample.txt', header=False, index=False, sep=' ')
	
if __name__ == '__main__':
	main()