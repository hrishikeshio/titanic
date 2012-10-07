from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import csv
import numpy as np
import math
from sklearn.svm import LinearSVC, SVC
from sklearn import cross_validation
from sklearn.preprocessing import *
from sklearn.grid_search import GridSearchCV
import copy
X_train=[]
y_train=[]
X_test=[]
y_test=[]

with open("train.csv",'rb') as f:
	reader=csv.reader(f)
	title= reader.next()
	titleidx=range(len(title))
	print titleidx
	print title

	row=copy.deepcopy(title)
	del  row[7] ,row[2], row[0]
	print row
	#exit()
	
		
	afterrow=copy.deepcopy(row)
	for row in reader:
		#print "===================================="
		#print title
		titleidx=range(len(title))
		#print titleidx
		#print afterrow
		##print row
		#==========GENDER
		row[3]=1 if row[3]=="male" else 0
		#==========FROM
		if row[10]=="C":
			row[10]=0
		elif row[10]=="S":
			row[10]=1
		elif row[10]=="Q":
			row[10]=2
		else:
			row[10]=1
		#=========AGE
		try:
			if float(row[4])<1:
				row[4]=1
			elif float(row[4])<5:
				row[4]=2
			elif float(row[4])<10:
				row[4]=3
			elif float(row[4])<20:
				row[4]=4
			elif float(row[4])<30:
				row[4]=5
			elif float(row[4])<40:
				row[4]=6
			elif float(row[4])<50:
				row[4]=6
			elif float(row[4])<60:
				row[4]=7
			elif float(row[4])>60:
				row[4]=7
			else:
				row[4]=5

		except:
			row[4]=3
		##print row
		#==========sibsp and parch
		try:
			row[5]=0 if float(row[5])==0 and float(row[6])==0  else 1
		except:
			row[5]=0
		try:
			row[6]=0 if float(row[6])==0 else 1
		except:
			row[6]=0
		#=========CABIN
		##print row
		cabin={'A':0, 'B':1, 'C':3, 'D':4, 'E':5 ,'F':6, 'G':7}
		try:
			row[9]=cabin[row[9][0].strip()]
		except:
			row[9]=0
		#print row
		#FARE
		try:
			#pass
			if float(row[8])<5:
				row[8]=1.
			elif float(row[8])<10:
				row[8]=1
			elif float(row[8])<15:
				row[8]=1
			elif float(row[8])<20:
				row[8]=2.
			elif float(row[8])<30:
				row[8]=3.
			elif float(row[8])<40:
				row[8]=3.
			elif float(row[8])<100:
				row[8]=3.
			elif float(row[8])<200:
				row[8]=3.
			elif float(row[8])<300:
				row[8]=3.
			elif float(row[8])>300:
				row[8]=3.
			else:
				row[8]=1.
		except:
				print row
				row[8]=1.
	#		rows[1]=round(math.ceil(rows[1]/10),3)
	
		#==========DELETE rows
		del row[7], row[5],row[4],row[2],row[1]
		#print row
		#exit()
		try:
			rows=[float(i) for i in row[1:]]
		except:
			pass
		#print rows

		#exit()
		#print rows
		X_train.append(rows)
		y_train.append(int(row[0]))

print len(X_train[0])
max1=0
max2=0
max3=0
for i in range(1,256):
	break
	idx=str(bin(i)).split("0b")[1]
	intidx=[]
	if len(idx)<8:
		idx="0"*(8-len(idx))+idx

	for j in range(len(idx)):
		if idx[j]=="1":
			intidx.append(j)
	#print idx
	#print intidx
	#print X_train[0]
	X_train_perm=[]
	for i in X_train:
		X_train_perm.append([i[j] for j in intidx])
	#print X_train_perm
	



	X_train_perm_scaled=scale(X_train_perm)
	clf=RandomForestClassifier()
	clf2=LogisticRegression(C=1)
	clf3=SVC(kernel='rbf')
	scores = cross_validation.cross_val_score(clf, X_train_perm, y_train, cv=5)
	if np.array(scores).mean()>max1:
		print "rf",np.array(scores).mean(),np.array(scores).std() 
		max1=np.array(scores).mean()
		comb1=intidx

	scores = cross_validation.cross_val_score(clf2, X_train_perm, y_train, cv=5)

	if np.array(scores).mean()>max2:
		print "logit",np.array(scores).mean(),np.array(scores).std() 
		max2=np.array(scores).mean()
		comb2=intidx
	scores = cross_validation.cross_val_score(clf3, X_train_perm_scaled, y_train, cv=5)

	if np.array(scores).mean()>max3:
		print "svm",np.array(scores).mean(),np.array(scores).std() 
		max3=np.array(scores).mean()#gammas = np.logspace(-6, 1, 10)
		comb3=intidx
#print comb1,comb2,comb3
#print max1,max2,max3
#clf = GridSearchCV(estimator=clf3, param_grid=dict(gamma=gammas),n_jobs=-1, cv=5)
#print cross_validation.cross_val_score(clf, X_train, y_train).mean()
#print X_train
X_train=scale(X_train)
exit()
clf=RandomForestClassifier()

clf.fit(X_train,y_train)
with open("test.csv","rb") as f:
	reader=csv.reader(f)
	reader.next()
	for row in reader:
		#print row
		row[3]=1 if row[3]=="male" else 0
		#print row
		if row[10]=="C":
			row[10]=0
		elif row[10]=="S":
			row[10]=1
		elif row[10]=="Q":
			row[10]=2
		else:
			row[10]=1
		#print row
		#=========AGE
		try:
			if float(row[4])<1:
				row[4]=1
			elif float(row[4])<5:
				row[4]=2
			elif float(row[4])<10:
				row[4]=3
			elif float(row[4])<20:
				row[4]=4
			elif float(row[4])<30:
				row[4]=5
			elif float(row[4])<40:
				row[4]=6
			elif float(row[4])<50:
				row[4]=6
			elif float(row[4])<60:
				row[4]=7
			elif float(row[4])>60:
				row[4]=7
			else:
				row[4]=5

		except:
			row[4]=3

		#print row
		#==========sibsp and parch
		try:
			row[5]=0 if float(row[5])==0 and float(row[6])==0  else 1
		except:
			row[5]=0
		try:
			row[6]=0 if float(row[6])==0 else 1
		except:
			row[6]=0
		#=========CABIN
		#print row
		cabin={'A':0, 'B':1, 'C':3, 'D':4, 'E':5 ,'F':6, 'G':7}
		try:
			row[9]=cabin[row[9][0].strip()]
		except:
			row[9]=0
		#print row
		#FARE
		try:
			#pass
			if float(row[8])<5:
				row[8]=1.
			elif float(row[8])<10:
				row[8]=1
			elif float(row[8])<15:
				row[8]=1
			elif float(row[8])<20:
				row[8]=2.
			elif float(row[8])<30:
				row[8]=3.
			elif float(row[8])<40:
				row[8]=3.
			elif float(row[8])<100:
				row[8]=3.
			elif float(row[8])<200:
				row[8]=3.
			elif float(row[8])<300:
				row[8]=3.
			elif float(row[8])>300:
				row[8]=3.
			else:
				row[8]=1.
		except:
				row[8]=1.
		#print row
	#		rows[1]=round(math.ceil(rows[1]/10),3)
	
		
		del row[7], row[5],row[4],row[2], row[1],row[0]
		
		print row
		try:
			rows=[float(i) for i in row]
		except:
			pass

	#	try:
	#		rows[2]=round(math.ceil(rows[2]/10),2)
	#		rows[1]=round(math.ceil(rows[1]/10),3)
		#except:
		#	pass
		#print rows
		#exit()
		#print rows
		X_test.append(rows)
X_test=scale(X_test)
#print X_train[0],X_test[0]
print len(X_train),len(y_train),len(X_test),len(y_train)
clf=RandomForestClassifier().fit(X_train,y_train)
clf2=LogisticRegression(C=1).fit(X_train,y_train)
clf3=SVC(kernel='rbf', probability=True).fit(X_train,y_train)

#preds=clf.predict_proba(X_test)
#preds2=clf2.predict_proba(X_test)

preds3=clf2.predict(X_test)
#totalpreds=(preds+preds2+preds3)/3

ans=[]
#for i in totalpreds:
#	ans.append(1 if i[0]<.5 else 0)
#print ans
with open("submissions/logitfeatsgridsearch.csv","wb") as f:
	writer=csv.writer(f)
	for row in preds3:
		writer.writerow([int(row)])

		