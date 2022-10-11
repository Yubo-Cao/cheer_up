(() => {
    let streaming = false;

    let video = document.getElementById('video');

    // take video stream from webcam
    navigator.mediaDevices.getUserMedia({video: true, audio: false})
        .then(function (stream) {
            video.srcObject = stream;
            video.play();
        })
        .catch(function (err) {
            console.log("An error occurred: " + err);
        });
    // crop video to 16:9
    video.addEventListener('canplay', function (e) {
        if (!streaming) {
            height = video.videoHeight / (video.videoWidth / width);

            video.setAttribute('width', width);
            video.setAttribute('height', height);
            streaming = true;
        }
    }, false);
})()