require('@tensorflow/tfjs-node-gpu');
const posenet = require('@tensorflow-models/posenet');
const ffmpeg = require('ffmpeg');
const ffmpeg2 = require('fluent-ffmpeg');
const fs = require('fs');
const _ = require('lodash');
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


async function analyze(vidLocation) {
    const net = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 16,
        inputResolution: 801,
        quantBytes: 4
    });
    const vid = await new ffmpeg(vidLocation);
    const skip = 1;
    let frames = await vid.fnExtractFrameToJPG('./frames', {
        every_n_frames: skip,
    });
    const lines = [
        ["leftShoulder", "leftElbow"],
        ["leftElbow", "leftWrist"],
        ["rightShoulder", "rightElbow"],
        ["rightElbow", "rightWrist"],
        ["leftShoulder", "leftHip"],
        ["rightShoulder", "rightHip"],
        ["leftShoulder", "rightShoulder"],
        ["leftHip", "rightHip"],
        ["leftHip", "leftKnee"],
        ["leftKnee", "leftAnkle"],
        ["rightHip", "rightKnee"],
        ["rightKnee", "rightAnkle"],
    ];
    const fps = vid.metadata.video.fps / skip;
    const {w, h} = vid.metadata.video.resolution;
    frames = _.sortBy(frames, f => parseInt(f.split('/').pop().split('.').shift().split('_').pop()));
    const allKeypoints = [];
    for (const frame of frames) {
        allKeypoints.push(_.keyBy((await getPose(frame, net, w, h)).keypoints, "part"));
    }
    const smoothed = [];
    const rangeWidth = 5;
    for (let i = 0; i < allKeypoints.length; i++) {
        const range = _.range(Math.max(i - rangeWidth, 0), Math.min(i + rangeWidth + 1, allKeypoints.length));
        const allParts = _.union(...range.map(j => Object.keys(allKeypoints[j])));
        let smoothedKeypoint = allParts.map(part => {
            const filteredRange = range.filter(j => part in allKeypoints[j]);
            const total = _.sum(filteredRange.map(j => allKeypoints[j][part].score));
            const weightedX = _.sum(filteredRange.map(j => allKeypoints[j][part].position.x * allKeypoints[j][part].score)) / total;
            const weightedY = _.sum(filteredRange.map(j => allKeypoints[j][part].position.y * allKeypoints[j][part].score)) / total;
            return {
                part,
                position: {
                    x: weightedX,
                    y: weightedY
                }
            }
        });
        smoothed.push(smoothedKeypoint)
    }

    smoothed.forEach((keypoints, i) => {
        const frame = frames[i];
        const canvas = createCanvas(w, h);
        const ctx = canvas.getContext('2d');
        loadImage(frame).then(image => {
            ctx.drawImage(image, 0, 0);
            keypoints.forEach(keypoint => {
                if (partToId[keypoint.part] >= 5) {
                    const {position: {x, y}} = keypoint;
                    ctx.beginPath();
                    ctx.arc(x, y, 5, 0, 2 * Math.PI, false);
                    ctx.fillStyle = 'black';
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = 'rgba(0,0,0,1)';
                    ctx.stroke();
                }
            });
            const pointsByPart = _.keyBy(keypoints, "part");
            const parts = Object.keys(pointsByPart);
            lines.forEach(([a, b]) => {
                if (parts.includes(a) && parts.includes(b)) {
                    const {position: {x: x1, y: y1}} = pointsByPart[a];
                    const {position: {x: x2, y: y2}} = pointsByPart[b];
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = 'rgba(0,0,0,1)';
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }
            });
            ctx.save();
            canvas.createJPEGStream().pipe(fs.createWriteStream(frame));
        });
    });
    ffmpeg2().input('frames/vid1_%01d.jpg').withOutputFPS(fps).save('output.mp4')
}

async function getPose(frame, net, w, h) {
    const canvas = createCanvas(w, h);
    const ctx = canvas.getContext('2d');
    const image = await loadImage(frame);
    ctx.drawImage(image, 0, 0);
    return await net.estimateSinglePose(ctx.canvas);
}


analyze('./vid1.MOV');
