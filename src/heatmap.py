#!/usr/bin/env python

# internal modules
import os
import sys
import signal
import glob
import platform
import subprocess
import argparse
import math

# external modules
import numpy as np
import pandas as pd
import seaborn as sbs
import matplotlib.pyplot as plt

def signal_handler(sig, frame):
    sys.exit(0)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# ChatGPT got this somewhat right, but not really
def needs_python_glob():
	# Test for known systems that have useful shells
	system = platform.system().lower()
	if system in ['linux','darwin'] or system.startswith('cygwin'):
		return False

	# Cygwin bash will report 'Windows' if using python.exe and not cygwin python.
	if system == 'windows':
		# detect CYGWIN. Typically SHELL is not defined unless CYGWIN
		return 'CYGWIN' not in os.environ and 'MSYSTEM' not in os.environ and 'SHELL' not in os.environ

	# Test if the shell expands the '*' wildcard by checking if it returns expanded files
	try:
		# In cygwin, if using python.exe, this will spawn cmd.exe, so make sure we catch cygwin above
		# This will expand '*' if the shell supports glob expansion, otherwise return as-is
		output = subprocess.check_output('echo *', shell=True, text=True)
		# If the output lists files (not "*"), we assume glob expansion worked
		return '*' in output
	except subprocess.CalledProcessError:
		return True

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
	lc['load'] = lc['load'].round(0)
	lc['rpm'] = (lc['rpm']/10).round(0) * 10
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
			if len(res.index) >= (args.min_samples, 1)[args.no_filter]: # ternary is (true, false)
				# size of cell, center to corner
				cellradius = distance(lv[0], rv[0], lv[1], rv[1])/2
				# ME7-style clamping: clamp coordinates to cell boundaries like the ECU does
				# Calculate distance to clamped point instead of cell center
				res['distance'] = res.apply(lambda row:
					distance(row.load, row.rpm,
						max(lv[0], min(lv[1], row.load)),  # clamp load to cell boundaries
						max(rv[0], min(rv[1], row.rpm))),   # clamp rpm to cell boundaries
					axis=1)
				# Weight is proportional to closeness to clamped point: distance = 0 has highest weight
				# Don't let weights get negative due to weird radius calcs
				# This matches ME7 behavior where out-of-bounds data is clamped to cell edges
				res['weight'] = np.maximum((cellradius-res.distance)/cellradius, 0.1)
				mean = res.fr.mean()
				median = res.fr.median()
				wa = np.average(res.fr, weights = res.weight)
				# pick median or weighted average
				data = (wa, median)[args.use_unweighted_median]
				frdata[lk][rk] = round(data, 3)
				if args.verbose>1:
					print(res.to_string())
				if args.verbose and abs(median-wa) >= 0.5:
					print(f"[{lk},{rk}] median:{round(median, 3)} mean:{round(mean, 3)} wa:{round(wa, 3)} diff:{round(median-wa, 3)}")
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

	parser.add_argument('-u', '--use-unweighted-median', action='store_true', help='use unweighted median instead of weighted average (default is weighted))')
	parser.add_argument('-c', '--continuous', action='store_true', help='show continuous instead of bucketed heatmap (default is bucketed)')

	parser.add_argument('-v', '--verbose', action='count', default=0)

	parser.add_argument('--text', action='store_true', help='output as text, no GUI')
	parser.add_argument('--csv', action='store_true', help='output as csv, no GUI')
	args = parser.parse_args()

	# do globbing ourselves if we need to
	if needs_python_glob():
		if args.verbose:
			print('Platform may not provide glob, using our own')
		filenames = []
		for file in args.filename:
			files = glob.glob(file)
			if len(files) == 0:
				eprint(f"No such file: '{file}'")
				return 1
			filenames += files
		args.filename = filenames

	dfa = []
	for file in args.filename:
		#print("Skipping info lines")
		try:
			with open(file, 'rb') as f:
				i=0
				for line in f:
					line = line.decode('unicode_escape').strip()
					# First header starts with TimeStamp
					if line.startswith('TimeStamp'):
						break
					#print(line)
					i = i + 1
		except Exception as error:
			eprint(error)
			return 1

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

	sbs.heatmap(heatmap, annot=(not args.continuous), center=0, cmap='PiYG_r', cbar_kws={'label': '% trim'}).invert_yaxis()

	plt.xlabel('RPM')
	plt.ylabel('Load')

	signal.signal(signal.SIGINT, signal_handler)
	plt.show()

def test_weighting_algorithm():
	"""Test cases for the weighting algorithm to verify inside/outside cell behavior"""
	print("Running weighting algorithm tests...")

	# Test data setup
	lv = [50, 100]  # load range
	rv = [2000, 3000]  # rpm range
	lk, rk = 75, 2500  # cell center
	cellradius = distance(lv[0], rv[0], lv[1], rv[1]) / 2

	def calculate_weight(row_load, row_rpm):
		# Check if point is inside cell boundaries
		inside_load = lv[0] <= row_load <= lv[1]
		inside_rpm = rv[0] <= row_rpm <= rv[1]

		if inside_load and inside_rpm:
			# Inside cell: distance to center
			dist = distance(row_load, row_rpm, lk, rk)
		else:
			# Outside cell: distance to clamped point
			clamped_load = max(lv[0], min(lv[1], row_load))
			clamped_rpm = max(rv[0], min(rv[1], row_rpm))
			dist = distance(row_load, row_rpm, clamped_load, clamped_rpm)

		weight = max((cellradius - dist) / cellradius, 0.1)
		return dist, weight

	# Test Case 1: Point at cell center
	dist, weight = calculate_weight(75, 2500)
	print(f"Center point (75, 2500): distance={dist:.2f}, weight={weight:.3f}")
	assert weight >= 0.9, f"Center point should have high weight, got {weight}"

	# Test Case 2: Point at cell edge
	dist, weight = calculate_weight(50, 2500)
	print(f"Edge point (50, 2500): distance={dist:.2f}, weight={weight:.3f}")

	# Test Case 3: Point outside cell (should clamp)
	dist, weight = calculate_weight(30, 2500)  # below load range
	print(f"Outside point (30, 2500): distance={dist:.2f}, weight={weight:.3f}")
	assert weight < 1.0, f"Outside point should have reduced weight, got {weight}"

	# Test Case 4: Point far outside cell
	dist, weight = calculate_weight(10, 1000)
	print(f"Far outside point (10, 1000): distance={dist:.2f}, weight={weight:.3f}")
	assert weight == 0.1, f"Far outside point should have minimum weight, got {weight}"

	# Test Case 5: Point at cell corner
	dist, weight = calculate_weight(100, 3000)
	print(f"Corner point (100, 3000): distance={dist:.2f}, weight={weight:.3f}")

	print("All weighting tests passed!")

def test_edge_cells():
	"""Test edge cell behavior with boundary clamping"""
	print("\nRunning edge cell tests...")

	# Test edge cell (minimum load)
	lv_edge = [0, 50]  # starts at 0
	rv_edge = [2000, 3000]
	lk_edge, rk_edge = 25, 2500  # center of edge cell
	cellradius = distance(lv_edge[0], rv_edge[0], lv_edge[1], rv_edge[1]) / 2

	def calculate_edge_weight(row_load, row_rpm):
		inside_load = lv_edge[0] <= row_load <= lv_edge[1]
		inside_rpm = rv_edge[0] <= row_rpm <= rv_edge[1]

		if inside_load and inside_rpm:
			dist = distance(row_load, row_rpm, lk_edge, rk_edge)
		else:
			clamped_load = max(lv_edge[0], min(lv_edge[1], row_load))
			clamped_rpm = max(rv_edge[0], min(rv_edge[1], row_rpm))
			dist = distance(row_load, row_rpm, clamped_load, clamped_rpm)

		weight = max((cellradius - dist) / cellradius, 0.1)
		return dist, weight

	# Test point below minimum load (should clamp to 0)
	dist, weight = calculate_edge_weight(-10, 2500)
	print(f"Below edge point (-10, 2500): distance={dist:.2f}, weight={weight:.3f}")
	assert dist > 0, f"Below edge point should have positive distance, got {dist}"

	# Test point at edge boundary
	dist, weight = calculate_edge_weight(0, 2500)
	print(f"At edge boundary (0, 2500): distance={dist:.2f}, weight={weight:.3f}")

	print("All edge cell tests passed!")

def test_different_cell_sizes():
	"""Test different cell sizes for proper weighting"""
	print("\nRunning cell size tests...")

	# Small cell
	small_lv = [75, 85]
	small_rv = [2500, 2600]
	small_center = (80, 2550)
	small_radius = distance(small_lv[0], small_rv[0], small_lv[1], small_rv[1]) / 2

	# Large cell
	large_lv = [50, 150]
	large_rv = [2000, 4000]
	large_center = (100, 3000)
	large_radius = distance(large_lv[0], large_rv[0], large_lv[1], large_rv[1]) / 2

	def calculate_cell_weight(row_load, row_rpm, lv, rv, center, radius):
		inside_load = lv[0] <= row_load <= lv[1]
		inside_rpm = rv[0] <= row_rpm <= rv[1]

		if inside_load and inside_rpm:
			dist = distance(row_load, row_rpm, center[0], center[1])
		else:
			clamped_load = max(lv[0], min(lv[1], row_load))
			clamped_rpm = max(rv[0], min(rv[1], row_rpm))
			dist = distance(row_load, row_rpm, clamped_load, clamped_rpm)

		weight = max((radius - dist) / radius, 0.1)
		return dist, weight

	# Test same point in different sized cells
	test_point = (80, 2550)

	# Small cell
	dist_small, weight_small = calculate_cell_weight(test_point[0], test_point[1],
		small_lv, small_rv, small_center, small_radius)
	print(f"Small cell point {test_point}: distance={dist_small:.2f}, weight={weight_small:.3f}")

	# Large cell
	dist_large, weight_large = calculate_cell_weight(test_point[0], test_point[1],
		large_lv, large_rv, large_center, large_radius)
	print(f"Large cell point {test_point}: distance={dist_large:.2f}, weight={weight_large:.3f}")

	print("All cell size tests passed!")

if __name__ == '__main__':
	# Run tests if --test flag is provided
	if len(sys.argv) > 1 and sys.argv[1] == '--test':
		test_weighting_algorithm()
		test_edge_cells()
		test_different_cell_sizes()
		print("\nAll tests completed successfully!")
		sys.exit(0)

	main()

# vim: ft=python noexpandtab
