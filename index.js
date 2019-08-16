require('@tensorflow/tfjs-node-gpu');
const posenet = require('@tensorflow-models/posenet');
const ffmpeg = require('ffmpeg');
const ffmpeg2 = require('fluent-ffmpeg');
const fs = require('fs');
const _ = require('lodash');
const {createCanvas, loadImage} = require('canvas');

async function analyze(vidLocation) {
    const net = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 16,
        inputResolution: 801,
        quantBytes: 4
    });
    const vid = await new ffmpeg(vidLocation);
    const skip = 3;
    const frames = await vid.fnExtractFrameToJPG('./frames', {
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
    for (const frame of frames) {
        const canvas = createCanvas(w, h);
        const ctx = canvas.getContext('2d');
        const image = await loadImage(frame);
        ctx.drawImage(image, 0, 0);
        const poses = await net.estimateMultiplePoses(ctx.canvas);
        poses.forEach(pose => {
            pose.keypoints.forEach(keypoint => {
                const {position: {x, y}} = keypoint;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI, false);
                ctx.fillStyle = 'black';
                ctx.fill();
                ctx.lineWidth = 2;
                ctx.strokeStyle = 'rgba(0,0,0,1)';
                ctx.stroke();
            });
            const pointsByPart = _.keyBy(pose.keypoints, "part");
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
            })
        });

        ctx.save();
        canvas.createJPEGStream().pipe(fs.createWriteStream(frame));
        console.log(frame);
    }
    ffmpeg2().input('frames/vid1_%01d.jpg').withOutputFPS(fps).save('output.mp4')

}

analyze('./vid1.MOV');
