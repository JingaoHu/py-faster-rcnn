
#-------------------------------------------------
# written by Jingao Hu
# function: split dataset to testset and trainset
#-------------------------------------------------

import pickle
import os

path="/home/jingao/vehicle-detection/VOCdevkit/VOC2007/ImageSets/Main/"

############################ generate the whole dataset 
dataset=[]
for i in xrange(64):
	for j in ['000','090','180','270']:
		if i<10 and i>0:dataset.append('0'+str(i)+j)
		if i>9:dataset.append(str(i)+j)


########################### define the test set by yourself 
dataset_test=['06000','07000','08000','12000','15000','17000','20000','30000','44000','51000','53000','60000','61000','62000']
f=open(path+"test.txt",'w')
for i in dataset_test:
	f.write(i+'\n')
	fi=file(path+'{}.txt'.format(i),'w')
	fi.write(i+'\n')
	fi.close
f.close()

f=file(path+'test.pkl','wb')
pickle.dump(dataset_test,f)
f.close()

########################### generate the trainval set 
dataset_trainval=[]

for i in dataset:
	notin=True
	for j in dataset_test:
		if i==j:
			notin=False
			break
	if notin:
		dataset_trainval.append(i)
		
f=open(path+"trainval.txt",'w')
for i in dataset_trainval:
	f.write(i+'\n')
f.close()

f=file(path+'trainval.pkl','wb')
pickle.dump(dataset_trainval,f)
f.close()

############################ the number of objects in each image
dataset_object={'01000':37,'02000':35,'03000':198,'04000':176,'05000':59,'06000':58,'07000':116,'08000':72,
'09000':103,'10000':201,'11000':123,'12000':38,'13000':61,'14000':57,'15000':37,'16000':78,
'17000':89,'18000':50,'19000':93,'20000':43,'21000':240,'22000':130,'23000':539,'24000':316,
'25000':245,'26000':127,'27000':127,'28000':41,'29000':174,'30000':45,'31000':133,'32000':113,
'33000':47,'34000':58,'35000':28,'36000':25,'37000':48,'38000':47,'39000':69,'40000':72,
'41000':167,'42000':57,'43000':38,'44000':48,'45000':126,'46000':110,'47000':135,'48000':92,
'49000':55,'50000':36,'51000':52,'52000':135,'53000':38,'54000':38,'55000':34,'56000':77,
'57000':40,'58000':134,'59000':117,'60000':91,'61000':214,'62000':277,'63000':166}

f=file(path+'num_object.pkl','wb')
pickle.dump(dataset_object,f)
f.close()

########################### generate car_test.txt
f=open(path+'temp.txt','wb')
for i in dataset:
	f.write(i+' -1\n')
f.close()

f=open(path+'car_test.txt','w')
f_temp=open(path+'temp.txt','r')
for line in f_temp:
	inTestset=False
	for j in dataset_test:
		if line[0:5]==j:
			inTestset=True
			break
	if inTestset:
		line=line.replace(' -1',' 1')
	f.write(line)

os.remove(path+'temp.txt')

