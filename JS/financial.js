const financialForm = document.getElementById("financialForm");


financialForm.addEventListener("submit", function (event) {

    event.preventDefault();


    const selectedCategories = Array.from(
        document.querySelectorAll(
            'input[name="specialCategory"]:checked'
        )
    ).map((item) => item.value);


    const financialData = {

        annualIncome:
            document.getElementById("income").value,

        annualRevenue:
            document.getElementById("revenue").value,

        currentInvestment:
            document.getElementById("investment").value,

        fundingRequired:
            document.getElementById("funding").value,

        existingLoan:
            document.querySelector(
                'input[name="existingLoan"]:checked'
            )?.value || "",

        specialCategories:
            selectedCategories

    };


    /*
        Temporary frontend storage.

        Later:
        profile + business + financial
        will be sent to FastAPI.
    */

    localStorage.setItem(
        "sakshamFinancial",
        JSON.stringify(financialData)
    );


    // Move to matching screen

    window.location.href = "matching.html";

});