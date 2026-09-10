/* =========================================
   VICHAR-SETU JAVASCRIPT
========================================= */


/* =========================================
   GLOBAL DATA
========================================= */

let selectedGoal = "";

let userData = {};


/* =========================================
   GOAL QUESTIONS
========================================= */

const goalQuestions = {

    business: [

        {
            id: "projectType",
            label: "What kind of business are you planning?",
            type: "select",
            options: [
                "Food Processing",
                "Tailoring",
                "Retail",
                "Manufacturing",
                "Service",
                "Other"
            ]
        },

        {
            id: "funding",
            label: "How much funding do you need?",
            type: "number",
            placeholder: "₹ Enter amount"
        },

        {
            id: "income",
            label: "Approximate annual family income",
            type: "number",
            placeholder: "₹ Enter annual income"
        }

    ],


    education: [

        {
            id: "educationLevel",
            label: "What are you studying?",
            type: "select",
            options: [
                "School",
                "Undergraduate",
                "Postgraduate",
                "Professional Course",
                "Vocational Course",
                "Other"
            ]
        },

        {
            id: "course",
            label: "What is your course or field?",
            type: "text",
            placeholder: "e.g. Computer Science"
        },

        {
            id: "educationCost",
            label: "Approximate education cost",
            type: "number",
            placeholder: "₹ Enter amount"
        }

    ],


    agriculture: [

        {
            id: "farmType",
            label: "What type of agricultural activity?",
            type: "select",
            options: [
                "Crop Farming",
                "Dairy",
                "Poultry",
                "Horticulture",
                "Fisheries",
                "Other"
            ]
        },

        {
            id: "landSize",
            label: "Approximate land size",
            type: "text",
            placeholder: "e.g. 2 acres"
        },

        {
            id: "agriFunding",
            label: "How much financial support do you need?",
            type: "number",
            placeholder: "₹ Enter amount"
        }

    ],


    skill: [

        {
            id: "skillType",
            label: "What skill do you want to develop?",
            type: "text",
            placeholder: "e.g. Computer skills"
        },

        {
            id: "trainingType",
            label: "Preferred training type?",
            type: "select",
            options: [
                "Online",
                "Offline",
                "Either"
            ]
        },

        {
            id: "trainingCost",
            label: "Approximate training cost",
            type: "number",
            placeholder: "₹ Enter amount"
        }

    ],


    housing: [

        {
            id: "housingNeed",
            label: "What kind of housing support do you need?",
            type: "select",
            options: [
                "New House",
                "House Repair",
                "Rent Support",
                "Basic Amenities",
                "Other"
            ]
        },

        {
            id: "income",
            label: "Approximate annual family income",
            type: "number",
            placeholder: "₹ Enter annual income"
        }

    ],


    financial: [

        {
            id: "financialNeed",
            label: "What type of financial assistance do you need?",
            type: "select",
            options: [
                "Loan",
                "Scholarship",
                "Grant",
                "Subsidy",
                "Emergency Assistance",
                "Other"
            ]
        },

        {
            id: "funding",
            label: "How much financial assistance do you need?",
            type: "number",
            placeholder: "₹ Enter amount"
        },

        {
            id: "income",
            label: "Approximate annual family income",
            type: "number",
            placeholder: "₹ Enter annual income"
        }

    ],


    other: [

        {
            id: "otherNeed",
            label: "What kind of support are you looking for?",
            type: "text",
            placeholder: "Describe what you need"
        },

        {
            id: "income",
            label: "Approximate annual family income",
            type: "number",
            placeholder: "₹ Enter annual income"
        }

    ]

};


/* =========================================
   GOAL NAMES
========================================= */

const goalNames = {

    business: "Starting or expanding a business",

    education: "Education / Studies",

    agriculture: "Agriculture / Farming",

    skill: "Skill development / Training",

    housing: "Housing / Basic needs",

    financial: "Financial assistance",

    other: "Something else"

};


/* =========================================
   DOM ELEMENTS
========================================= */

const steps = document.querySelectorAll(".step");

const progressBar = document.getElementById("progressBar");

const stepText = document.getElementById("stepText");


/* =========================================
   GO TO STEP
========================================= */

function goToStep(stepNumber) {

    steps.forEach(step => {

        step.classList.remove("active");

    });


    document
        .getElementById(`step${stepNumber}`)
        .classList.add("active");


    updateProgress(stepNumber);

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


/* =========================================
   PROGRESS BAR
========================================= */

function updateProgress(stepNumber) {

    const progress = (stepNumber / 5) * 100;

    progressBar.style.width = `${progress}%`;

    stepText.textContent = `Step ${stepNumber} of 5`;

}


/* =========================================
   GOAL SELECTION
========================================= */

document.querySelectorAll(".goal-card").forEach(card => {

    card.addEventListener("click", function () {

        selectedGoal = this.dataset.goal;

        console.log("Selected Goal:", selectedGoal);

        goToStep(2);

    });

});


/* =========================================
   GUIDED FORM BUTTON
========================================= */

document
    .getElementById("guidedBtn")
    .addEventListener("click", function () {

        showGuidedForm();

        goToStep(3);

    });


/* =========================================
   NATURAL LANGUAGE BUTTON
========================================= */

document
    .getElementById("languageBtn")
    .addEventListener("click", function () {

        document
            .getElementById("guidedForm")
            .classList.add("hidden");

        document
            .getElementById("languageForm")
            .classList.remove("hidden");

        goToStep(3);

    });


/* =========================================
   SHOW GUIDED FORM
========================================= */

function showGuidedForm() {

    document
        .getElementById("guidedForm")
        .classList.remove("hidden");

    document
        .getElementById("languageForm")
        .classList.add("hidden");


    const questionsContainer =
        document.getElementById("dynamicQuestions");


    questionsContainer.innerHTML = "";


    const questions =
        goalQuestions[selectedGoal];


    document.getElementById("formTitle").textContent =
        `Tell us about your ${goalNames[selectedGoal].toLowerCase()}`;


    questions.forEach(question => {

        const group = document.createElement("div");

        group.className = "input-group";


        const label = document.createElement("label");

        label.textContent = question.label;

        group.appendChild(label);


        let input;


        /* SELECT */

        if (question.type === "select") {

            input = document.createElement("select");

            input.id = question.id;

            input.required = true;


            const defaultOption =
                document.createElement("option");

            defaultOption.value = "";

            defaultOption.textContent =
                "Select an option";

            input.appendChild(defaultOption);


            question.options.forEach(option => {

                const optionElement =
                    document.createElement("option");

                optionElement.value = option;

                optionElement.textContent = option;

                input.appendChild(optionElement);

            });

        }


        /* INPUT */

        else {

            input = document.createElement("input");

            input.type = question.type;

            input.id = question.id;

            input.placeholder =
                question.placeholder || "";

            input.required = true;

        }


        group.appendChild(input);

        questionsContainer.appendChild(group);

    });

}


/* =========================================
   FORM SUBMISSION
========================================= */

document
    .getElementById("detailsForm")
    .addEventListener("submit", function (event) {

        event.preventDefault();


        collectGuidedData();

        displayReview();

        goToStep(4);

    });


/* =========================================
   COLLECT GUIDED DATA
========================================= */

function collectGuidedData() {

    userData = {

        purpose: goalNames[selectedGoal],

        age: document.getElementById("age").value,

        gender: document.getElementById("gender").value,

        category: document.getElementById("category").value,

        state: document.getElementById("state").value,

        district: document.getElementById("district").value,

        occupation: document.getElementById("occupation").value

    };


    const questions =
        goalQuestions[selectedGoal];


    questions.forEach(question => {

        const element =
            document.getElementById(question.id);

        if (element) {

            userData[question.id] =
                element.value;

        }

    });


    console.log("User Data:", userData);

}


/* =========================================
   NATURAL LANGUAGE EXTRACTION
========================================= */

document
    .getElementById("extractBtn")
    .addEventListener("click", function () {

        const text =
            document
                .getElementById("userDescription")
                .value
                .trim();


        if (text === "") {

            alert(
                "Please describe your situation first."
            );

            return;

        }


        extractInformation(text);

        displayReview();

        goToStep(4);

    });


/* =========================================
   AI EXTRACTION DEMO
========================================= */

function extractInformation(text) {

    /*
        IMPORTANT:

        This is a FRONTEND DEMO.

        In the real project,
        this function should call your
        backend AI/NLP API.

        Example:

        fetch("/api/extract-profile", {
            method: "POST",
            body: JSON.stringify({
                text: text
            })
        })

    */


    userData = {

        purpose: goalNames[selectedGoal],

        age: extractAge(text),

        gender: extractGender(text),

        category: extractCategory(text),

        state: extractState(text),

        district: extractDistrict(text),

        occupation: extractOccupation(text),

        projectType:
            extractProjectType(text),

        income:
            extractIncome(text),

        funding:
            extractFunding(text)

    };


    console.log(
        "AI Extracted Data:",
        userData
    );

}


/* =========================================
   EXTRACTION FUNCTIONS
========================================= */

function extractAge(text) {

    const match =
        text.match(/\b(\d{2})\b/);

    return match
        ? match[1]
        : "Not provided";

}


function extractGender(text) {

    const lower =
        text.toLowerCase();


    if (
        lower.includes("woman") ||
        lower.includes("female") ||
        lower.includes("girl")
    ) {

        return "Female";

    }


    if (
        lower.includes("man") ||
        lower.includes("male") ||
        lower.includes("boy")
    ) {

        return "Male";

    }


    return "Not provided";

}


function extractCategory(text) {

    const lower =
        text.toLowerCase();


    if (lower.includes("sc")) {

        return "SC";

    }


    if (lower.includes("st")) {

        return "ST";

    }


    if (lower.includes("obc")) {

        return "OBC";

    }


    return "Not provided";

}


function extractState(text) {

    const lower =
        text.toLowerCase();


    if (
        lower.includes("uttar pradesh") ||
        lower.includes("up")
    ) {

        return "Uttar Pradesh";

    }


    if (lower.includes("bihar")) {

        return "Bihar";

    }


    if (lower.includes("rajasthan")) {

        return "Rajasthan";

    }


    if (lower.includes("maharashtra")) {

        return "Maharashtra";

    }


    return "Not provided";

}


function extractDistrict(text) {

    const lower =
        text.toLowerCase();


    const districts = [

        "lucknow",
        "kanpur",
        "varanasi",
        "agra",
        "prayagraj",
        "meerut",
        "gorakhpur"

    ];


    for (const district of districts) {

        if (lower.includes(district)) {

            return capitalize(district);

        }

    }


    return "Not provided";

}


function extractOccupation(text) {

    const lower =
        text.toLowerCase();


    if (lower.includes("student")) {

        return "Student";

    }


    if (lower.includes("farmer")) {

        return "Farmer";

    }


    if (lower.includes("tailor")) {

        return "Tailor";

    }


    return "Not provided";

}


function extractProjectType(text) {

    const lower =
        text.toLowerCase();


    if (
        lower.includes("food") ||
        lower.includes("food-processing")
    ) {

        return "Food Processing";

    }


    if (lower.includes("tailoring")) {

        return "Tailoring";

    }


    if (lower.includes("retail")) {

        return "Retail";

    }


    if (lower.includes("manufacturing")) {

        return "Manufacturing";

    }


    return "Not provided";

}


function extractIncome(text) {

    const match =
        text.match(
            /(?:income|family income).*?₹?\s?([\d.]+)\s*(lakh|lakhs|k)?/i
        );


    if (!match) {

        return "Not provided";

    }


    let amount =
        parseFloat(match[1]);


    if (
        match[2] &&
        match[2].toLowerCase().startsWith("l")
    ) {

        amount = amount * 100000;

    }


    return "₹" + amount.toLocaleString("en-IN");

}


function extractFunding(text) {

    const match =
        text.match(
            /(?:need|funding|fund|require).*?₹?\s?([\d.]+)\s*(lakh|lakhs|k)?/i
        );


    if (!match) {

        return "Not provided";

    }


    let amount =
        parseFloat(match[1]);


    if (
        match[2] &&
        match[2].toLowerCase().startsWith("l")
    ) {

        amount = amount * 100000;

    }


    return "₹" + amount.toLocaleString("en-IN");

}


/* =========================================
   CAPITALIZE
========================================= */

function capitalize(word) {

    return word.charAt(0).toUpperCase()
        + word.slice(1);

}


/* =========================================
   REVIEW SCREEN
========================================= */

function displayReview() {

    const review =
        document.getElementById("reviewData");


    review.innerHTML = "";


    const labels = {

        purpose: "Purpose",

        projectType: "Project Type",

        funding: "Funding Needed",

        income: "Family Income",

        educationLevel: "Education Level",

        course: "Course",

        educationCost: "Education Cost",

        farmType: "Agricultural Activity",

        landSize: "Land Size",

        skillType: "Skill",

        trainingType: "Training Type",

        trainingCost: "Training Cost",

        housingNeed: "Housing Need",

        financialNeed: "Financial Need",

        otherNeed: "Support Needed",

        age: "Age",

        gender: "Gender",

        category: "Category",

        state: "State",

        district: "District",

        occupation: "Occupation"

    };


    Object.entries(userData).forEach(
        ([key, value]) => {

            if (
                value &&
                value !== "Not provided"
            ) {

                const item =
                    document.createElement("div");

                item.className =
                    "review-item";


                const label =
                    document.createElement("span");

                label.textContent =
                    labels[key] || key;


                const data =
                    document.createElement("strong");

                data.textContent =
                    value;


                item.appendChild(label);

                item.appendChild(data);

                review.appendChild(item);

            }

        }
    );

}


/* =========================================
   CONTINUE TO ELIGIBILITY
========================================= */

document
    .getElementById("continueBtn")
    .addEventListener("click", function () {

        goToStep(5);

        startEligibilityProcess();

    });


/* =========================================
   ELIGIBILITY PROCESS
========================================= */

function startEligibilityProcess() {

    const loadingTitle =
        document.getElementById("loadingTitle");

    const loadingText =
        document.getElementById("loadingText");


    const eligibilityItem =
        document.getElementById("eligibilityItem");

    const matchingItem =
        document.getElementById("matchingItem");

    const rankingItem =
        document.getElementById("rankingItem");


    /* STEP 1 */

    loadingTitle.textContent =
        "Checking your eligibility";

    loadingText.textContent =
        "We're analyzing your information...";


    setTimeout(() => {

        eligibilityItem.classList.add(
            "active"
        );

        eligibilityItem
            .querySelector("span")
            .textContent = "✓";


    }, 1200);


    /* STEP 2 */

    setTimeout(() => {

        matchingItem.classList.add(
            "active"
        );

        matchingItem
            .querySelector("span")
            .textContent = "✓";


        loadingTitle.textContent =
            "Matching relevant schemes";

        loadingText.textContent =
            "Finding support options that fit your profile...";


    }, 2400);


    /* STEP 3 */

    setTimeout(() => {

        rankingItem.classList.add(
            "active"
        );

        rankingItem
            .querySelector("span")
            .textContent = "✓";


        loadingTitle.textContent =
            "Ranking recommendations";

        loadingText.textContent =
            "Prioritizing the most relevant options...";


    }, 3600);


    /* RESULTS */

    setTimeout(() => {

        document
            .getElementById("loadingScreen")
            .classList.add("hidden");

        document
            .getElementById("resultsScreen")
            .classList.remove("hidden");


    }, 4800);

}