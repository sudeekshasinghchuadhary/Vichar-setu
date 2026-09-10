/* =========================
   SAKSHAM AI
   FINANCIAL ANALYSIS
========================= */

const financialData =
    JSON.parse(localStorage.getItem("sakshamFinancial")) || {};


/* =========================
   HELPER
========================= */

function formatMoney(value) {

    const number = Number(value) || 0;

    return "₹" + number.toLocaleString("en-IN");
}


/* =========================
   GET SAVED VALUES
========================= */

const income =
    Number(financialData.annualIncome) || 400000;

const revenue =
    Number(financialData.annualRevenue) || 250000;

const investment =
    Number(financialData.currentInvestment) || 100000;

const funding =
    Number(financialData.fundingRequired) || 500000;


/* =========================
   UPDATE SUMMARY CARDS
========================= */

document.getElementById("incomeValue").textContent =
    formatMoney(income);

document.getElementById("revenueValue").textContent =
    formatMoney(revenue);

document.getElementById("investmentValue").textContent =
    formatMoney(investment);

document.getElementById("fundingValue").textContent =
    formatMoney(funding);


/* =========================
   FUNDING GAP
========================= */

const fundingGap =
    Math.max(funding - investment, 0);


/*
   Find the funding gap element
   and update it.
*/

const fundingRows =
    document.querySelectorAll(".funding-row");

if (fundingRows.length >= 2) {

    fundingRows[1]
        .querySelector("strong")
        .textContent = formatMoney(fundingGap);
}


/* =========================
   FINANCIAL READINESS
========================= */

/*
   Frontend prototype calculation.

   This is NOT the final AI/eligibility
   calculation. Backend can replace it later.
*/

let readinessScore = 70;


/* Self-investment effect */

if (investment >= funding * 0.30) {

    readinessScore += 10;

} else if (investment >= funding * 0.15) {

    readinessScore += 5;
}


/* Revenue effect */

if (revenue > 0) {

    readinessScore += 5;
}


/* Large funding gap */

if (fundingGap > income) {

    readinessScore -= 8;
}


readinessScore =
    Math.max(40, Math.min(95, readinessScore));


/* Update score */

document.getElementById("readinessScore").textContent =
    readinessScore;


/* =========================
   STATUS
========================= */

const statusPill =
    document.querySelector(".status-pill");

const scoreHeading =
    document.querySelector(".score-area h3");

const scoreDescription =
    document.querySelector(".score-area p");


if (readinessScore >= 80) {

    statusPill.textContent = "Strong";

    scoreHeading.textContent =
        "Strong financial position";

    scoreDescription.textContent =
        "Your current financial profile shows a relatively strong base for pursuing suitable opportunities.";

} else if (readinessScore >= 65) {

    statusPill.textContent = "Moderate";

    scoreHeading.textContent =
        "Good starting position";

    scoreDescription.textContent =
        "Your profile shows a reasonable financial base, with additional funding potentially needed for growth.";

} else {

    statusPill.textContent = "Needs support";

    scoreHeading.textContent =
        "Additional support may help";

    scoreDescription.textContent =
        "Your current profile suggests that funding or other financial support could be useful.";
}


/* =========================
   ANIMATION
========================= */

document.querySelectorAll(".summary-card").forEach((card, index) => {

    card.style.opacity = "0";
    card.style.transform = "translateY(15px)";

    setTimeout(() => {

        card.style.transition =
            "all 0.4s ease";

        card.style.opacity = "1";
        card.style.transform = "translateY(0)";

    }, index * 100);

});


console.log("Saksham AI Financial Analysis loaded.");