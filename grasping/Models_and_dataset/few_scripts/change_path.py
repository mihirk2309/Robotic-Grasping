f = open('/home/pranay/robotic-grasping-cornell/dataset/grasp/ImageSets/test.txt', "r")
lines = f.readlines()
f1 = open('/home/pranay/robotic-grasping-cornell/dataset/grasp/ImageSets/test1.txt', "a")
count = 0
for line in lines:
    print(line[0:23])        # print(line[78:])     
    count+=1
    print(line[27:27])
    f1.write(line[0:23])    # f1.write(line[78:])
    f1.write(line[27:])
print(count)
