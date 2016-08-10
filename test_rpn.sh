rm output/faster_rcnn_alt_opt/voc_2007_trainval/*.pkl

./experiments/scripts/test_rpn.sh 0 ZF pascal_voc

python tools/proposals_merge.py
