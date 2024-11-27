// Show the loader when the DOM content has loaded
document.addEventListener("DOMContentLoaded", showLoader);


window.addEventListener("load", function() {
    setTimeout(hideLoader, 3000); 
});


document.getElementById("generateBtn").addEventListener("click", generateImage);

// Function to handle image generation
function generateImage() {
    var generateBtn = document.getElementById("generateBtn");
    generateBtn.disabled = true; // Disable the button to prevent multiple submissions
    generateBtn.value = 'Loading...'; // Change button text to indicate loading
    showLoader(); // Show the loader

    var textInput = document.getElementById("textInput").value;
    if (!textInput) {
        alert('Please enter some text.');
        resetButton();
        hideLoader();
        return; // Exit the function if no input is provided
    }

    var query = encodeURIComponent(textInput);
    var apiUrl1 = `http://172.18.43.5:5001/${query}`;
   
    var apiUrl2 = `http://172.18.43.5:5001/${query}`;

    var imageContainer1 = document.getElementById('imageContainer1');
    imageContainer1.onload = function() {
        hideLoader(); // Hide the loader when the image has loaded
        resetButton(); // Reset the button and clear the input field
    };
    imageContainer1.onerror = function() {
        hideLoader(); // Hide the loader if there's an error
        resetButton(); // Reset the button and clear the input field
        alert('Failed to load image.'); // Show an error message
    };
    imageContainer1.src = apiUrl1; // Set the image source to start loading
    var delayInMilliseconds = 1000; //1 second

    setTimeout(function() {
  //your code to be executed after 1 second
        }, delayInMilliseconds);
    var imageContainer2 = document.getElementById('imageContainer2');
    imageContainer2.onload = function() {
        hideLoader(); // Hide the loader when the image has loaded
        resetButton(); // Reset the button and clear the input field
    };
    imageContainer2.onerror = function() {
        hideLoader(); // Hide the loader if there's an error
        resetButton(); // Reset the button and clear the input field
        alert('Failed to load image.'); // Show an error message
    };
    imageContainer2.src = apiUrl2; // Set the image source to start loading
}

// Function to show the loader
function showLoader() {
    var loader = document.querySelector(".loader");
    loader.classList.remove("loader--hidden"); // Remove the hidden class to show the loader
    loader.style.position = 'absolute'; // Position the loader relative to the image container
}

// Function to hide the loader
function hideLoader() {
    var loader = document.querySelector(".loader");
    loader.classList.add("loader--hidden"); // Add the hidden class to hide the loader
}

// Function to reset the 'Generate Image' button and clear the input field
function resetButton() {
    var generateBtn = document.getElementById("generateBtn");
    generateBtn.disabled = false; // Re-enable the button
    generateBtn.value = 'Generate Image'; // Reset button text
    var textInput = document.getElementById("textInput");
    textInput.value = ''; // Clear the input field
}

// Function to handle image load error
function imageLoadError() {
    alert('Failed to load image.'); // Show an error message
}
