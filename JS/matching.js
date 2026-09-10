const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");

const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const step3 = document.getElementById("step3");

const mainStatus = document.getElementById("mainStatus");


function completeStep(step) {

    step.classList.remove("active");
    step.classList.add("completed");

}


function activateStep(step) {

    step.classList.add("active");

}


/*
    Frontend prototype simulation.

    Later this will be replaced by:
    POST /match
*/

let progress = 0;

const matchingInterval = setInterval(() => {

    progress += 2;

    progressBar.style.width = progress + "%";

    progressText.textContent = progress + "%";


    // Step 1

    if (progress >= 25 && progress < 45) {

        completeStep(step1);

        activateStep(step2);

    }


    // Step 2

    if (progress >= 55 && progress < 80) {

        completeStep(step2);

        activateStep(step3);

    }


    // Step 3

    if (progress >= 80) {

        completeStep(step3);

    }


    // Finished

    if (progress >= 100) {

        clearInterval(matchingInterval);

        mainStatus.textContent = "COMPLETE";

        mainStatus.style.background = "#eaf4c9";
        mainStatus.style.color = "#171717";


        setTimeout(() => {

            window.location.href = "results.html";

        }, 700);

    }

}, 65);