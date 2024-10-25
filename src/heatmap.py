#!/usr/bin/env python

import os
import argparse
import math
import numpy as np
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

def continuous_heatmap(lc):
	lc['load'] = lc['load'].round(1)
	heatmap = lc.pivot_table(index='load', columns='rpm', values='fr')
	return heatmap.dropna(how='all', axis=0).dropna(how='all', axis=1)

def distance(x0, y0, x1, y1):
	return math.hypot(x1-x0, y1-y0)

def quantized_heatmap(lc, args):
	# create bucket ranges
	rpms = calc_ranges([720, 1000, 1240, 1520, 2000, 2520, 3000, 3520, 4000, 4520, 5000, 5520, 6000, 6520], 10000)
	loads = calc_ranges([9.75, 20.25, 30, 39.75, 50.25, 60, 69.75, 80.25, 90, 99.75, 110.25, 120, 140.25, 159.75], 300)
	#loads = calc_ranges([9.75, 20.25, 30, 39.75, 50.25, 60, 69.75, 80.25, 90, 99.75, 110.25, 140.25, 150, 168])

	# sort lambda control (fr/frm) into buckets
	frdata={}
	for lk,lv in loads.items():
		frdata[lk]={}
		for rk,rv in rpms.items():
			query = f"{rv[0]} <= @lc.rpm <= {rv[1]} & {lv[0]} <= @lc.load <= {lv[1]}"
			res = lc.query(query).copy()
			if len(res.index) >= (args.min_samples, 1)[args.no_filter]: # ternary is (false, true)
				# size of cell, center to corner
				cellradius = distance(lv[0], rv[0], lv[1], rv[1])/2
				# find distance to cell center
				# FIXME: this is not right for cells at edges of map, but we have to weight data off the map somehow
				res['distance'] = res.apply(lambda row: distance(row.load, row.rpm, lk, rk), axis=1)
				# Weight is proportional to closeness to center: distance = 0 has highest weight
				# Don't let weights get negative due to weird radius calcs
				res['weight'] = np.maximum((cellradius-res.distance)/cellradius, 0.1)
				mean = res.fr.mean()
				wa = np.average(res.fr, weights = res.weight)
				# pick mean or weighted average
				data = (wa, mean)[args.use_unweighted_mean]
				frdata[lk][rk] = round(data, 3)
				if args.verbose>1:
					print(res.to_string())
				if args.verbose:
					print(f"[{lk},{rk}] mean:{round(mean, 3)} wa:{round(wa, 3)} diff:{round(abs(mean-wa), 3)}")
	heatmap = pd.DataFrame(frdata). \
		dropna(how='all', axis=0). \
		dropna(how='all', axis=1). \
		sort_index(axis=0, ascending=True). \
		sort_index(axis=1, ascending=True).T

	return heatmap

def main():
	parser = argparse.ArgumentParser(description='Create trim heatmap for KFKHFM based on fr/frm datalog')
	parser.add_argument('filename', default=['log.csv'], nargs='*', help='csv files(s) to parse (log.csv)')
	parser.add_argument('-w', '--window', type=int, default=10, help='number of sequential rows to detect constant rpm/load (10)')

	parser.add_argument('-l', '--load-filter', type=float, default=10, help='change in load which is still "constant" load (10)')
	parser.add_argument('-r', '--rpm-filter', type=int, default=100, help='change in RPM which is still "constant" RPM (100)')
	parser.add_argument('-m', '--maf-filter', type=float, default=10, help='change in MAF which is still "constant" MAF (10)')

	parser.add_argument('-n', '--no-filter', action='store_true', help='disable filter (default is enabled)')

	parser.add_argument('-s', '--min-samples', type=int, default=10, help='minimum number of samples required to generate a cell (10)')
	parser.add_argument('-f', '--use-fr', action='store_true', help='use "fr" instead of "frm" (default is frm)')

	parser.add_argument('-u', '--use-unweighted-mean', action='store_true', help='use unweighted mean instead of weighted average (default is weighted))')
	parser.add_argument('-c', '--continuous', action='store_true', help='show continuous instead of bucketed heatmap (default is bucketed)')

	parser.add_argument('-v', '--verbose', action='count', default=0)

	parser.add_argument('--text', action='store_true', help='output as text, no GUI')
	parser.add_argument('--csv', action='store_true', help='output as csv, no GUI')
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
		dfa.append(pd.read_csv(file, sep=',', encoding='unicode_escape', skiprows=i, header=[0,1,2], skipinitialspace=True))

	# concat all the dfas into a single frame
	df = pd.concat(dfa, axis=0, ignore_index=True)
	# strip junk out of columns
	df.rename(str.strip, axis='columns', inplace=True)

	# pick rl or frm_w
	whichrl = ('rl', 'rl_w')['rl_w' in df]
	# pick nmot or nmot_w
	whichnmot = ('nmot', 'nmot_w')['nmot_w' in df]
	# pick frm or fr
	whichfr = ('frm', 'fr')[args.use_fr or 'frm_w' not in df]

	rows = [whichnmot, whichrl, whichfr + '_w', 'mshfm_w']

	if whichfr + '2_w' in df:
		rows.append(whichfr + '2_w')

	if args.verbose:
		print(f"using {rows} from log")

	# grab only the things we need, rename rl/nmot/mshfm to load/rpm/maf
	lc = df[rows]. \
		rename(columns={whichrl:'load'}). \
		rename(columns={whichnmot:'rpm'}). \
		rename(columns={'mshfm_w':'maf'})

	# flatten index
	lc.columns = lc.columns.get_level_values(0)

	# extract lambda control
	if whichfr + '2_w' in df:
		# average bank1 and bank2, convert to %
		lc['fr'] = (lc[[whichfr+'_w',whichfr+'2_w']].mean(axis=1) - 1) * 100
	else:
		# convert to %
		lc['fr'] = (lc[whichfr+'_w'] - 1) * 100

	# set up filter source data
	lc['rpm_delta'] = lc.rpm.rolling(window=args.window).apply(lambda x: x.max() - x.min())
	lc['load_delta'] = lc.load.rolling(window=args.window).apply(lambda x: x.max() - x.min())
	lc['maf_delta'] = lc.maf.rolling(window=args.window).apply(lambda x: x.max() - x.min())

	if not args.no_filter:
		# tag rows we want to use based on source data
		lc['use'] = \
			lc.rpm_delta.notnull()  & (lc.rpm_delta  <= args.rpm_filter)  & \
			lc.load_delta.notnull() & (lc.load_delta <= args.load_filter) & \
			lc.maf_delta.notnull()  & (lc.maf_delta  <= args.maf_filter)

		#print(lc[(abs(lc.fr)>5) & (lc.rpm_delta > 0)].to_string())

		# throw out data that moves too much
		#print(lc.to_string())
		lc = lc[(lc['use'])]
		#print(lc.to_string())

	heatmap = (quantized_heatmap(lc, args), continuous_heatmap(lc))[args.continuous]

	if args.text:
		print(heatmap.sort_index(axis=0, ascending=False).to_string())

	if args.csv:
		print(heatmap.to_csv())

	if args.text or args.csv:
		return 0

	fig, ax = plt.subplots()
	if len(args.filename) == 1:
		fig.canvas.manager.set_window_title(os.path.basename(args.filename[0]))
	else:
		fig.canvas.manager.set_window_title('Trim heatmap')

	ax.tick_params(labelbottom=False,labeltop=True)
	ax.xaxis.set_ticks_position('top')
	ax.xaxis.set_label_position('top')

	sbs.heatmap(heatmap, annot=(not args.continuous), center=0, cmap='PiYG', cbar_kws={'label': '% trim'}).invert_yaxis()

	plt.xlabel('RPM')
	plt.ylabel('Load')
	plt.show()

if __name__ == '__main__':
	main()

# vim: ft=python noexpandtab
