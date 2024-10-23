#!/usr/bin/env python

import argparse
import pandas as pd
import seaborn as sbs
import matplotlib.pyplot as plt

def calc_ranges(ary, max):
	buckets = {}
	for i,v in enumerate(ary):
		if i == 0:
			# very start
			buckets[v] = [0, (ary[i]+ary[i+1])/2-1]
		elif i != len(ary)-1:
			buckets[v] = [(ary[i-1]+ary[i])/2, (ary[i]+ary[i+1])/2-1]
		else:
			# very end
			buckets[v] = [(ary[i-1]+ary[i])/2, max]
	return buckets

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', default='log.csv', nargs='*')
	parser.add_argument('-f', '--use-fr', action='store_true')
	parser.add_argument('-l', '--load-filter', default=10)
	parser.add_argument('-m', '--min-samples', default=10)
	parser.add_argument('-r', '--rpm-filter', default=80)
	parser.add_argument('-w', '--window', default=5)
	parser.add_argument('-v', '--verbose', action='store_true')
	args = parser.parse_args()

	dfa = []
	for file in args.filename:
		#print("Skipping info lines")
		with open(file, 'rb') as f:
			i=0
			for line in f:
				line = line.decode('unicode_escape').strip()
				# First header starts with TimeStamp
				if line.startswith('TimeStamp'):
					break
				#print(line)
				i = i + 1

		#print(f"Loading headers from line {i+1}")
		#print(f"Loading data from line {i+4}")

		# Three header lines
		dfa.append(pd.read_csv(file, sep=",", encoding='unicode_escape', skiprows=i, header=[0,1,2], skipinitialspace=True))

	# concat all the dfas into a single frame
	df = pd.concat(dfa, axis=0, ignore_index=True)
	# strip junk out of columns
	df.rename(str.strip, axis='columns', inplace=True)

	# normalize column names
	if 'rl_w' in df:
		lc = df[["nmot", "rl_w", "fr_w", "fr2_w", "frm_w", "frm2_w"]]
		lc = lc.rename(columns={'rl_w':'load'})
	else:
		lc = df[["nmot", "rl", "fr_w", "fr2_w", "frm_w", "frm2_w"]]
		lc = lc.rename(columns={'rl':'load'})

	lc.rename(columns={'nmot':'rpm'}, inplace=True)

	# flatten index
	lc.columns = lc.columns.get_level_values(0)

	# extract lambda control
	lc['rpm_delta'] = lc.rpm.rolling(window=args.window).apply(lambda x: x.max() - x.min())
	lc['load_delta'] = lc.load.rolling(window=args.window).apply(lambda x: x.max() - x.min())

	# throw out data that moves too much
	#print(lc.to_string())
	lc = lc[(lc.rpm_delta < args.rpm_filter) & (lc.load_delta < args.load_filter)]
	#print(lc.to_string())

	# convert to %
	lc.loc[:, ["fr_w", "fr2_w", "frm_w", "frm2_w"]] -= 1.0
	lc.loc[:, ["fr_w", "fr2_w", "frm_w", "frm2_w"]] *= 100

	# average bank1 and bank2
	lc['fr'] = lc[['fr_w','fr2_w']].mean(axis=1)
	lc['frm'] = lc[['frm_w','frm2_w']].mean(axis=1)


	# create bucket ranges
	rpms = calc_ranges([720, 1000, 1240, 1520, 2000, 2520, 3000, 3520, 4000, 4520, 5000, 5520, 6000, 6520], 10000)
	loads = calc_ranges([9.75, 20.25, 30, 39.75, 50.25, 60, 69.75, 80.25, 90, 99.75, 110.25, 120, 140.25, 159.75], 300)
	#loads = calc_ranges([9.75, 20.25, 30, 39.75, 50.25, 60, 69.75, 80.25, 90, 99.75, 110.25, 140.25, 150, 168])

	# sort into buckets
	lcdata={}
	for lk,lv in loads.items():
		lcdata[lk]={}
		for rk,rv in rpms.items():
			query = f"{rv[0]} <= @lc.rpm <= {rv[1]} & {lv[0]} <= @lc.load <= {lv[1]}"
			res = lc.query(query)
			if len(res.index) >= args.min_samples:
				if args.use_fr:
					lcdata[lk][rk] = round(float(res.fr.mean()), 2)
				else:
					lcdata[lk][rk] = round(float(res.frm.mean()), 2)
				if args.verbose:
					print(lk, rk, lcdata[lk][rk])

	fig, ax = plt.subplots()
	ax.tick_params(labelbottom=False,labeltop=True)
	ax.xaxis.set_ticks_position('top')
	ax.xaxis.set_label_position('top')

	heatmap = pd.DataFrame(lcdata).T
	heatmap.sort_index(axis=0, ascending=True, inplace=True)
	heatmap.sort_index(axis=1, ascending=True, inplace=True)
	sbs.heatmap(heatmap, annot=True, center=0, cmap='PiYG', cbar_kws={'label': '% trim'}).invert_yaxis()

	plt.xlabel("RPM")
	plt.ylabel("Load")
	plt.show()

if __name__ == "__main__":
	main()
