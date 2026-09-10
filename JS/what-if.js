/* =========================
   SAKSHAM AI — WHAT IF
========================= */

const incomeInput = document.getElementById("income");
const incomeRange = document.getElementById("incomeRange");

const simulateBtn = document.getElementById("simulateBtn");

const oldScore = document.getElementById("oldScore");
const newScore = document.getElementById("newScore");

const resultText = document.getElementById("resultText");
const resultStatus = document.getElementById("resultStatus");


/* =========================
   SYNC INPUT + SLIDER
========================= */

incomeRange.addEventListener("input", () => {

    incomeInput.value = incomeRange.value;

});


incomeInput.addEventListener("input", () => {

    let value = Number(incomeInput.value);

    if (value < 100000) {
        value = 100000;
    }

    if (value > 1000000) {
        value = 1000000;
    }

    incomeRange.value = value;

});


/* =========================
   SIMULATE MATCH
========================= */

simulateBtn.addEventListener("click", () => {

    const income = Number(incomeInput.value);

    const currentScore = 68;

    let simulatedScore = currentScore;


    /*
       Frontend-only simulation.

       Lower income → stronger simulated match
       Higher income → slightly lower match
    */

    if (income <= 250000) {

        simulatedScore = 82;

    } else if (income <= 350000) {

        simulatedScore = 76;

    } else if (income <= 500000) {

        simulatedScore = 68;

    } else if (income <= 750000) {

        simulatedScore = 62;

    } else {

        simulatedScore = 55;

    }


    oldScore.textContent =
        `${currentScore}%`;

    newScore.textContent =
        `${simulatedScore}%`;


    /* =========================
       RESULT MESSAGE
    ========================= */

    if (simulatedScore > currentScore) {

        resultText.textContent =
            "This scenario could make your profile a stronger potential match.";

        resultStatus.textContent =
            "Potential match improvement";

    } else if (simulatedScore < currentScore) {

        resultText.textContent =
            "This scenario could reduce your potential match for this opportunity.";

        resultStatus.textContent =
            "Lower potential match";

    } else {

        resultText.textContent =
            "This change does not significantly affect the simulated match.";

        resultStatus.textContent =
            "No major change";

    }


    /* =========================
       ANIMATION
    ========================= */

    newScore.style.transform = "scale(1.15)";

    setTimeout(() => {

        newScore.style.transition =
            "transform 0.3s ease";

        newScore.style.transform =
            "scale(1)";

    }, 100);

});