let model;
let videoWidth, videoHeight;
let ctx, canvas;
const log = document.querySelector("#array");
const VIDEO_WIDTH = 600;
const VIDEO_HEIGHT = 350;
const knnClassifier = ml5.KNNClassifier();

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

let fingerLookupIndices = {
  thumb: [0, 1, 2, 3, 4],
  indexFinger: [0, 5, 6, 7, 8],
  middleFinger: [0, 9, 10, 11, 12],
  ringFinger: [0, 13, 14, 15, 16],
  pinky: [0, 17, 18, 19, 20],
};

let detectedPose = false;
let resultsArray = [];

const btnTrainYT = document.getElementById("btnTrainYT");
const btnTrainGoogle = document.getElementById("btnTrainGoogle");
const btnClassify = document.getElementById("btnClassify");

btnTrainYT.addEventListener("click", () => {
  learn("https://www.youtube.com/");
});
btnTrainGoogle.addEventListener("click", () => {
  learn("https://www.google.com/");
});
btnClassify.addEventListener("click", () => {
  classify();
});

async function main() {
  model = await handpose.load();
  const video = await setupCamera();
  video.play();
  startLandmarkDetection(video);
}

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Webcam not available");
  }
  const video = document.getElementById("video");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: VIDEO_WIDTH,
      height: VIDEO_HEIGHT,
    },
  });
  video.srcObject = stream;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function startLandmarkDetection(video) {
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

  canvas = document.getElementById("output");

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  ctx = canvas.getContext("2d");

  video.width = videoWidth;
  video.height = videoHeight;

  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = "red";
  ctx.fillStyle = "red";

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  predictLandmarks();
}

async function predictLandmarks() {
  ctx.drawImage(video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);
  const predictions = await model.estimateHands(video);
  if (predictions.length > 0) {
    const result = predictions[0].landmarks;
    drawKeypoints(ctx, result, predictions[0].annotations);
    resultsArray = result.flat();
    detectedPose = true;
  } else {
    detectedPose = false;
  }
  requestAnimationFrame(predictLandmarks);
}

function learn(label) {
  knnClassifier.addExample(resultsArray, label);
}

function classify() {
  setInterval(() => {
    if (detectedPose) {
      knnClassifier.classify(resultsArray, (err, result) => {
        console.log(result.label);
        window.open(result.label, "_blank");
      });
    }
  }, 10000);
}

function drawKeypoints(ctx, keypoints) {
  const keypointsArray = keypoints;

  for (let i = 0; i < keypointsArray.length; i++) {
    const y = keypointsArray[i][0];
    const x = keypointsArray[i][1];
    drawPoint(ctx, x - 2, y - 2, 3);
  }

  const fingers = Object.keys(fingerLookupIndices);
  for (let i = 0; i < fingers.length; i++) {
    const finger = fingers[i];
    const points = fingerLookupIndices[finger].map((idx) => keypoints[idx]);
    drawPath(ctx, points, false);
  }
}

function drawPoint(ctx, y, x, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}
function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}
main();
