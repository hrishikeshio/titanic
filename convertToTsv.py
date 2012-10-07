import csv

with open("train.csv","rb") as f:
	with open("train.tsv","wb") as o:
		ass=csv.reader(f)
		out=csv.writer(o, delimiter='	')
		for i in ass:
			
			out.writerow(i)