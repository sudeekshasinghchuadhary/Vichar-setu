const schemeData = {

    1: {
        name: "Pradhan Mantri MUDRA Yojana",

        type: "Business Funding",

        description:
            "Collateral-free institutional credit for micro and small businesses looking to start or expand their venture.",

        match: 94,

        status: "✓ Strong Match",

        support: "Loans up to ₹10 lakh"
    },


    2: {
        name: "Prime Minister's Employment Generation Programme",

        type: "Business + Employment",

        description:
            "Credit-linked subsidy programme designed to help entrepreneurs establish new micro enterprises.",

        match: 89,

        status: "✓ Strong Match",

        support: "Margin money subsidy"
    },


    3: {
        name: "Stand-Up India",

        type: "Entrepreneurship",

        description:
            "Bank loans to support eligible greenfield enterprises in manufacturing, services or trading.",

        match: 84,

        status: "✓ Good Match",

        support: "₹10 lakh – ₹1 crore"
    },


    4: {
        name: "Credit Guarantee Fund Scheme",

        type: "Credit Support",

        description:
            "Credit guarantee support that helps eligible micro and small enterprises access institutional finance.",

        match: 79,

        status: "✓ Good Match",

        support: "Credit guarantee"
    },


    5: {
        name: "National SC-ST Hub",

        type: "Entrepreneurship",

        description:
            "Support for eligible entrepreneurs through capacity building, market access and business assistance.",

        match: 68,

        status: "Near Match",

        support: "Business support"
    },


    6: {
        name: "PMEGP — Expansion Support",

        type: "Business Growth",

        description:
            "Support opportunities for eligible entrepreneurs seeking assistance for enterprise development and growth.",

        match: 61,

        status: "Near Match",

        support: "Financial assistance"
    }

};



/* =========================
   GET SCHEME ID
========================= */

const params = new URLSearchParams(
    window.location.search
);

const schemeId = params.get("id") || "1";

const scheme = schemeData[schemeId] || schemeData[1];



/* =========================
   UPDATE PAGE
========================= */

document.getElementById("schemeName").textContent =
    scheme.name;

document.getElementById("schemeType").textContent =
    scheme.type;

document.getElementById("schemeDescription").textContent =
    scheme.description;

document.getElementById("matchScore").textContent =
    scheme.match + "%";

document.getElementById("schemeStatus").textContent =
    scheme.status;

document.getElementById("support").textContent =
    scheme.support;



/* =========================
   APPLICATION BUTTON
========================= */

document.getElementById("applyButton")
    .addEventListener("click", function () {

        alert(
            "Application pathway will be available in the next prototype phase."
        );

    });