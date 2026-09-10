const schemes = [

    {
        id: 1,

        name: "Pradhan Mantri MUDRA Yojana",

        type: "Business Funding",

        description:
            "Collateral-free institutional credit for micro and small businesses looking to start or expand their venture.",

        match: 94,

        status: "eligible",

        statusText: "✓ Strong Match",

        support: "Loans up to ₹10 lakh"
    },


    {
        id: 2,

        name: "Prime Minister's Employment Generation Programme",

        type: "Business + Employment",

        description:
            "Credit-linked subsidy programme designed to help entrepreneurs establish new micro enterprises.",

        match: 89,

        status: "eligible",

        statusText: "✓ Strong Match",

        support: "Margin money subsidy"
    },


    {
        id: 3,

        name: "Stand-Up India",

        type: "Entrepreneurship",

        description:
            "Bank loans to support greenfield enterprises in manufacturing, services or trading sectors.",

        match: 84,

        status: "eligible",

        statusText: "✓ Good Match",

        support: "₹10 lakh – ₹1 crore"
    },


    {
        id: 4,

        name: "Credit Guarantee Fund Scheme",

        type: "Credit Support",

        description:
            "Credit guarantee support that helps eligible micro and small enterprises access institutional finance.",

        match: 79,

        status: "eligible",

        statusText: "✓ Good Match",

        support: "Credit guarantee"
    },


    {
        id: 5,

        name: "National SC-ST Hub",

        type: "Entrepreneurship",

        description:
            "Support for eligible SC/ST entrepreneurs through capacity building, market access and business assistance.",

        match: 68,

        status: "near",

        statusText: "Near Match",

        support: "Business support"
    },


    {
        id: 6,

        name: "PMEGP — Expansion Support",

        type: "Business Growth",

        description:
            "Support opportunities for eligible entrepreneurs seeking assistance for enterprise development and growth.",

        match: 61,

        status: "near",

        statusText: "Near Match",

        support: "Financial assistance"
    }

];



const schemeGrid = document.getElementById("schemeGrid");

const filterButtons =
    document.querySelectorAll(".filter");

const sortSelect =
    document.getElementById("sort");



/* =========================
   DISPLAY SCHEMES
========================= */

function displaySchemes(data) {

    schemeGrid.innerHTML = "";


    if (data.length === 0) {

        schemeGrid.innerHTML = `
            <div class="no-results">
                No schemes found for this filter.
            </div>
        `;

        return;
    }


    data.forEach((scheme) => {

        const card = document.createElement("article");

        card.className = "scheme-card";


        card.innerHTML = `

            <div class="card-top">

                <span class="scheme-type">
                    ${scheme.type}
                </span>

                <div class="match-score">

                    <strong>
                        ${scheme.match}%
                    </strong>

                    <span>
                        Match score
                    </span>

                </div>

            </div>


            <h2>
                ${scheme.name}
            </h2>


            <p class="description">
                ${scheme.description}
            </p>


            <div class="eligibility ${scheme.status === "near" ? "near" : ""}">

                ${scheme.statusText}

            </div>


            <div class="card-bottom">

                <div>

                    <span class="support-label">
                        POTENTIAL SUPPORT
                    </span>

                    <strong class="support-value">
                        ${scheme.support}
                    </strong>

                </div>


                <a
                    href="scheme-details.html?id=${scheme.id}"
                    class="details-link"
                >
                    View Details →
                </a>

            </div>

        `;


        schemeGrid.appendChild(card);

    });

}


/* =========================
   FILTER
========================= */

function applyFilter(filter) {

    if (filter === "near") {

        displaySchemes(
            schemes.filter(
                scheme => scheme.status === "near"
            )
        );

        // Add Near-Miss link below the cards
        setTimeout(() => {

            const nearLink = document.createElement("div");

            nearLink.className = "near-miss-link";

            nearLink.innerHTML = `
                <a href="near-miss.html">
                    Explore why you're close →
                </a>
            `;

            schemeGrid.appendChild(nearLink);

        }, 50);

        return;
    }


    if (filter === "all") {

        displaySchemes(schemes);

        return;
    }


    const filtered =
        schemes.filter(
            scheme => scheme.status === filter
        );

    displaySchemes(filtered);
}


filterButtons.forEach((button) => {

    button.addEventListener("click", () => {

        filterButtons.forEach((btn) => {

            btn.classList.remove("active");

        });


        button.classList.add("active");


        applyFilter(
            button.dataset.filter
        );

    });

});



/* =========================
   SORT
========================= */

sortSelect.addEventListener("change", () => {

    const sorted = [...schemes];


    if (sortSelect.value === "match") {

        sorted.sort(
            (a, b) => b.match - a.match
        );

    }


    if (sortSelect.value === "amount") {

        sorted.sort(
            (a, b) => b.match - a.match
        );

    }


    displaySchemes(sorted);

});



/* =========================
   PROFILE SUMMARY
========================= */

const profile =
    JSON.parse(
        localStorage.getItem("sakshamProfile")
    );


if (profile && profile.name) {

    document.getElementById("resultSummary").textContent =
        `${profile.name}, based on your profile we've identified schemes that could be relevant to your goals.`;

}



/* =========================
   INITIAL LOAD
========================= */

displaySchemes(schemes);