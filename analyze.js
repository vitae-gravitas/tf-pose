// Run file with node analyze.js <video file name> <skip rate> <...add other params if needed, you will need to add params in analyze>
require('@tensorflow/tfjs-node-gpu');
const posenet = require('@tensorflow-models/posenet');
const ffmpeg = require('ffmpeg');
const fs = require('fs');
const {createCanvas, loadImage} = require('canvas');
const _ = require('lodash');

const partToId = {
    nose: 0,
    leftEye: 1,
    rightEye: 2,
    leftEar: 3,
    rightEar: 4,
    leftShoulder: 5,
    rightShoulder: 6,
    leftElbow: 7,
    rightElbow: 8,
    leftWrist: 9,
    rightWrist: 10,
    leftHip: 11,
    rightHip: 12,
    leftKnee: 13,
    rightKnee: 14,
    leftAnkle: 15,
    rightAnkle: 16
};

/**
 *
 * @param vidLocation
 * @param skip Frame skip rate
 * Other params can be added as needed.
 */
async function analyze(vidLocation, skip = 3) {
    const net = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 16,
        inputResolution: 801,
        quantBytes: 4
    });
    const vid = await new ffmpeg(vidLocation);
    const frames = await vid.fnExtractFrameToJPG('./frames', {
        every_n_frames: skip,
    });
    const fps = vid.metadata.video.fps / skip;
    const {w, h} = vid.metadata.video.resolution;
    const analysisResult = {
        video: {
            fps,
            w, h
        },
    };

    let keypoints = _.keyBy((await getPose(frames.shift(), net, w, h)).keypoints, "part");
    const allPoses = [keypoints];
    for (const frame of frames) {
        const k2 = _.keyBy((await getPose(frame, net, w, h)).keypoints, "part");
        const allParts = _.union(Object.keys(keypoints), Object.keys(k2));
        keypoints = allParts.map(part => {
            if (!(part in keypoints)) {
                return k2[part];
            } else if (!(part in k2)) {
                return keypoints[part];
            } else {
                const hist = keypoints[part];
                const newKeypoint = k2[part];
                const newX = (newKeypoint.position.x * newKeypoint.score) + hist.position.x * (1-newKeypoint.score);
                const newY = (newKeypoint.position.y * newKeypoint.score) + hist.position.y * (1-newKeypoint.score);
                return {
                    part,
                    position: {
                        x: newX,
                        y: newY
                    }
                }
            }
        });
        keypoints = _.keyBy(keypoints, "part");
        allPoses.push(keypoints);
    }
    analysisResult.poses = allPoses;
    fs.writeFileSync(`${vidLocation}.json`, JSON.stringify(analysisResult), 'utf8');
}

async function getPose(frame, net, w, h) {
    const canvas = createCanvas(w, h);
    const ctx = canvas.getContext('2d');
    const image = await loadImage(frame);
    ctx.drawImage(image, 0, 0);
    return await net.estimateSinglePose(ctx.canvas);
}

const args = process.argv.slice(2);
analyze(...args);
