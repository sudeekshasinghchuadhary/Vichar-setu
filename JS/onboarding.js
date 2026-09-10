const profileForm = document.getElementById("profileForm");

profileForm.addEventListener("submit", function (event) {

    event.preventDefault();

    // Collect profile data
    const profileData = {

        name: document.getElementById("name").value,

        age: document.getElementById("age").value,

        gender: document.getElementById("gender").value,

        occupation: document.getElementById("occupation").value,

        state: document.getElementById("state").value,

        category: document.querySelector(
            'input[name="category"]:checked'
        )?.value || ""

    };


    // Temporary local storage.
    // Later this can be replaced by FastAPI.

    localStorage.setItem(
        "sakshamProfile",
        JSON.stringify(profileData)
    );


    // Continue to business details

    window.location.href = "business-details.html";

});