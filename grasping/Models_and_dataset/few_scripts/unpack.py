import pickle

with open('/home/pranay/robotic-grasping-cornell/few_scripts/grasp_gt_rectdb.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)