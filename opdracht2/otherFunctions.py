import sys

def progress(count, total, suffix='', printamount = 100):
	"""
	Prints a progress bar in the terminal, indicating the progress in percentage\n
	count (int):
		Progress to date (iterable of a for loop for example)\n
	total (int):
		Max length of the work to be done (number of loops on for loop for example)\n
	suffix = '' (str):
		Suffix printed behind the progress bar\n
		printamoutn = 1000 (int): the amount of times the progress is printed
	"""
	if (count % int(total / printamount)) == 0:
		bar_len = 60
		filled_len = int(round(bar_len * count / float(total)))
		
		percents = round(100.0 * count / float(total), 1)
		bar = '=' * filled_len + '-' * (bar_len - filled_len)
		
		sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
		sys.stdout.flush()
