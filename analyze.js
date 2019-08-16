// Run file with node analyze.js <video file name> <skip rate> <...add other params if needed, you will need to add params in analyze>
require('@tensorflow/tfjs-node-gpu');
const posenet = require('@tensorflow-models/posenet');
const ffmpeg = require('ffmpeg');
const fs = require('fs');
const {createCanvas, loadImage} = require('canvas');

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
    const allPoses = [];
    for (const frame of frames) {
        const canvas = createCanvas(w, h);
        const ctx = canvas.getContext('2d');
        const image = await loadImage(frame);
        ctx.drawImage(image, 0, 0);
        let poses = await net.estimateMultiplePoses(ctx.canvas);
        poses = poses.map(pose => // Serialize so JSON isn't huge
            [
                pose.score,
                pose.keypoints.map(keypoint =>
                    [partToId[keypoint.part], Math.round(keypoint.score, 3), Math.round(keypoint.position.x, 6), Math.round(keypoint.position.y, 6)])
            ]);
        allPoses.push(poses);
        console.log(frame);
    }
    analysisResult.poses = allPoses;
    fs.writeFileSync(`${vidLocation}.json`, JSON.stringify(analysisResult), 'utf8');
}

const args = process.argv.slice(2);
analyze(...args);
