/* =========================
   SAKSHAM AI
   SCHEME COMPARISON
========================= */


/* =========================
   SCHEME DATA
========================= */

const schemes = [

    {
        id: 1,
        name: "PM MUDRA Yojana",
        category: "Business Funding",
        match: 94,
        support: "Loans up to ₹10 lakh",
        application: "Online / Bank",
        type: "Credit Support",
        bestFor: "Small businesses",
        status: "Eligible"
    },

    {
        id: 2,
        name: "PMEGP",
        category: "Employment + Business",
        match: 89,
        support: "Margin Money Support",
        application: "Online",
        type: "Entrepreneurship",
        bestFor: "New enterprises",
        status: "Eligible"
    },

    {
        id: 3,
        name: "Stand-Up India",
        category: "Entrepreneurship",
        match: 84,
        support: "₹10 lakh – ₹1 crore",
        application: "Bank",
        type: "Business Loan",
        bestFor: "Entrepreneurs",
        status: "Eligible"
    }

];


/* =========================
   GET SCHEME ID
========================= */

const params = new URLSearchParams(window.location.search);

const selectedIds =
    params.get("ids");


/*
   Example URL:

   scheme-comparison.html?ids=1,2,3

   If no IDs are provided,
   show the default top 3 schemes.
*/

let selectedSchemes = schemes;


if (selectedIds) {

    const ids = selectedIds
        .split(",")
        .map(Number);

    selectedSchemes =
        schemes.filter(scheme =>
            ids.includes(scheme.id)
        );

}


/* =========================
   UPDATE SCHEME COLUMNS
========================= */

const schemeColumns =
    document.querySelectorAll(".scheme-column");


selectedSchemes.forEach((scheme, index) => {

    if (!schemeColumns[index]) {
        return;
    }

    const column =
        schemeColumns[index];


    /* Scheme name */

    const title =
        column.querySelector(".scheme-header h2");

    if (title) {
        title.textContent = scheme.name;
    }


    /* Category */

    const category =
        column.querySelector(".scheme-header p");

    if (category) {
        category.textContent = scheme.category;
    }


    /* Match score */

    const score =
        column.querySelector(".compare-value.score");

    if (score) {
        score.textContent =
            `${scheme.match}%`;
    }


    /* Other comparison values */

    const values =
        column.querySelectorAll(".compare-value");


    if (values[1]) {
        values[1].textContent =
            scheme.support;
    }


    if (values[2]) {
        values[2].textContent =
            scheme.application;
    }


    if (values[3]) {
        values[3].textContent =
            scheme.type;
    }


    if (values[4]) {
        values[4].textContent =
            scheme.bestFor;
    }


    /* Eligibility */

    if (values[5]) {

        values[5].innerHTML = `
            <span class="eligible">
                ✓ ${scheme.status}
            </span>
        `;

    }

});


/* =========================
   UPDATE RECOMMENDATION
========================= */

if (selectedSchemes.length > 0) {

    const bestScheme =
        [...selectedSchemes]
            .sort((a, b) => b.match - a.match)[0];


    const recommendationTitle =
        document.querySelector(".recommendation h2");

    const recommendationText =
        document.querySelector(".recommendation p");


    if (recommendationTitle) {

        recommendationTitle.textContent =
            `${bestScheme.name} is currently your strongest match.`;

    }


    if (recommendationText) {

        recommendationText.textContent =
            `Based on the simulated profile match, it has the highest match score among the compared options.`;

    }


    /* Update View Scheme button */

    const viewButton =
        document.querySelector(".primary-btn");


    if (viewButton) {

        viewButton.href =
            `scheme_details.html?id=${bestScheme.id}`;

    }

}


/* =========================
   ANIMATION
========================= */

document
    .querySelectorAll(".scheme-column")
    .forEach((column, index) => {

        column.style.opacity = "0";
        column.style.transform = "translateY(15px)";


        setTimeout(() => {

            column.style.transition =
                "all 0.4s ease";

            column.style.opacity = "1";
            column.style.transform =
                "translateY(0)";

        }, index * 120);

    });


console.log(
    "Saksham AI Scheme Comparison loaded."
);