import json

json_filename = "test0_cones.json"
with open(json_filename) as f:
        json_data = json.load(f)
for i in range(len(json_data[1]['log_data'])):
    detections = json_data[1]['log_data'][i]['detections']
    # print(len(detections))
    for j in range(len(detections)):  
        print(detections[j][2][0:2])
    # print(detections)
    # for j in range(len(detections)):
    #     a = detections[0][2]
    #     for i in range(len(a)):
    #         print(a[i])
    # print(detections)

a = [['blue', '47.99', [399, 284, 415, 347]], ['blue', '74.17', [157, 182, 167, 204]], ['blue', '93.31', [284, 197, 298, 227]], ['blue', '94.0', 
[317, 218, 338, 260]], ['blue', '95.18', [138, 191, 152, 220]], ['blue', '95.84', [150, 186, 161, 211]], ['blue', '96.49', [62, 223, 88, 279]], ['blue', '97.12', [277, 191, 288, 216]], ['blue', '97.61', [343, 234, 371, 292]], ['blue', '98.34', [165, 179, 174, 199]], ['blue', '98.65', [297, 206, 314, 239]], ['blue', '99.1', [122, 197, 139, 231]], ['blue', '99.42', [99, 207, 121, 248]], ['blue', '99.61', [4, 252, 40, 330]]]

