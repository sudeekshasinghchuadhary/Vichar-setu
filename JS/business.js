const businessStatus = document.querySelectorAll(
    'input[name="businessStatus"]'
);

const businessFields = document.getElementById("businessFields");


businessStatus.forEach((status) => {

    status.addEventListener("change", function () {

        if (this.value === "existing" || this.value === "starting") {
            businessFields.classList.add("show");
        } else {
            businessFields.classList.remove("show");
        }

    });

});


const businessForm = document.getElementById("businessForm");


businessForm.addEventListener("submit", function (event) {

    event.preventDefault();


    const selectedGoals = Array.from(
        document.querySelectorAll(
            'input[name="goal"]:checked'
        )
    ).map((goal) => goal.value);


    const businessData = {

        businessStatus: document.querySelector(
            'input[name="businessStatus"]:checked'
        )?.value || "",

        businessName:
            document.getElementById("businessName").value,

        businessSector:
            document.getElementById("businessSector").value,

        businessStage:
            document.getElementById("businessStage").value,

        employees:
            document.getElementById("employees").value,

        goals: selectedGoals

    };


    // Temporary frontend storage

    localStorage.setItem(
        "sakshamBusiness",
        JSON.stringify(businessData)
    );


    // Next step

    window.location.href = "financial-details.html";

});