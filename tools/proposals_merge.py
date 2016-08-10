import pickle
import os

print 'Proposals merging ...'
proposals_path="/home/jingao/vehicle-detection/py-faster-rcnn/output/faster_rcnn_alt_opt/voc_2007_trainval/"
proposals_files=os.listdir(proposals_path)

result_file="/home/jingao/vehicle-detection/VOCdevkit/results/VOC2007/Main/comp3_det_test_car.txt"

re=open(result_file,'w')

for proposals_file in proposals_files:
	proposals=pickle.load(open(proposals_path+proposals_file,'rw'))
	for i in xrange(len(proposals[0])):
		re.write('{} {} {} {} {} {}\n'.format(proposals_file[0:5],proposals[0][i][4],proposals[0][i][0],proposals[0][i][1],proposals[0][i][2],proposals[0][i][3]))
	
re.close()

print 'Done!'


