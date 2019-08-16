import json
import os

idToPart = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow',
            'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle',
            'rightAnkle']


def deserialize(pose):
    return {
        "score": pose[0],
        "keypoints": [{
            "part": idToPart[keypoint[0]],
            "score": keypoint[1],
            "position": {
                keypoint[2],
                keypoint[3]
            }
        } for keypoint in pose[1]]
    }


file = 'vid1.MOV'
os.system('node analyze.js ' + file)

with open(file + '.json') as f:
    analysis = json.load(f)
    video_meta = analysis["video"]
    # Schema is: time evolution of poses, poses is an array (each entry is one person)
    poseEvolution = analysis["poses"]
    poseEvolution = list(map(lambda poses: list(map(deserialize, poses)), poseEvolution))
