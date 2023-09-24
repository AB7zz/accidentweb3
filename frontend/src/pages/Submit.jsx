import React, { useState, useRef } from 'react';
import { useStateContext } from '../context/StateContext';

function Submit() {
  const { address, apiUrl } = useStateContext();
  console.log(apiUrl)
  const videoRef = useRef(null);
  const inputRef = useRef(null);
  const canvasRef = useRef(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [capturedLocation, setCapturedLocation] = useState({});

  const captureLocation = () => {
    //
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        setCapturedLocation({ latitude, longitude });
      },
      (error) => {
        console.error('Error capturing location:', error);
      }
    );
  };

  const startCamera = async () => {
    try {
      captureLocation();
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
    } catch (error) {
      console.error('Error starting the camera:', error);
    }
  };

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert the canvas image to a Data URL
    const imageDataUrl = canvas.toDataURL('image/jpeg');

    inputRef.current.value = imageDataUrl;

    console.log(imageDataUrl)


    setCapturedImage(imageDataUrl);

    video.srcObject.getTracks().forEach((track) => track.stop());
  };



  const handleSubmit = () => {
    console.log("submit clicked")
    if (capturedImage && capturedLocation) {
      const dataToSend = {
        snapShot: capturedImage,
        _loc: capturedLocation, 
        _user: address
      };
      // console.log(formData)
      fetch('http://localhost:3000/userAddsAccident', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(dataToSend),
      })
        .then((response) => {
          if (response.ok) {
            console.log("response",response)
            console.log('Image and location sent successfully.');
          } else {
            // Handle error
            console.error('Failed to send image and location.');
          }
        })
        .catch((error) => {
          console.error('Error sending data:', error);
        });
    } else {
      console.error('Image or location not captured.');
    }
  };

  return (
    <div className="mt-10 mb-10" style={{ width: '529px', marginTop: '122px' }}>
      <h1 className="font-bold text-xl">Submit Accident</h1>
      <form>
        <div className="mb-6">
          <label htmlFor="Snapshot" className="block mb-2 text-sm font-medium text-gray-900">
            Live Snapshot
          </label>
          <div className="relative">
            <video ref={videoRef} className="border border-gray-300 rounded-lg" autoPlay playsInline></video>
            <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
          </div>
        </div>
        <div className="mb-6">
          <input
            ref={inputRef}
            type="text"
            id="snapshot"
            className="hidden border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:border-gray-600 dark:placeholder-gray-400 dark:focus:ring-blue-500 dark:focus:border-blue-500"
            readOnly
          />
        </div>
        {capturedImage && (
          <div className="mb-6">
            <img src={capturedImage} alt="Captured" className="max-w-full h-auto" />
          </div>
        )}
        <div className='flex flex-row gap-5'>
        <button
          type="button"
          onClick={startCamera}
          className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm w-full sm:w-auto px-5 py-2.5 text-center dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
        >
          Start Camera
        </button>
        <button
          type="button"
          onClick={captureImage}
          className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm w-full sm:w-auto px-5 py-2.5 text-center dark-bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800 mt-2"
        >
          Capture Image
        </button>
        <button
          type="button"
          onClick={handleSubmit}
          className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm w-full sm:w-auto px-5 py-2.5 text-center dark-bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800 mt-2"
        >
          Submit
        </button>
        </div>
      </form>
    </div>
  );
}

export default Submit;
