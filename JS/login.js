const passwordInput = document.getElementById("password");
const togglePassword = document.getElementById("togglePassword");

togglePassword.addEventListener("click", function () {

    if (passwordInput.type === "password") {
        passwordInput.type = "text";
        togglePassword.textContent = "◉";
    } else {
        passwordInput.type = "password";
        togglePassword.textContent = "◉";
    }

});


const loginForm = document.getElementById("loginForm");

loginForm.addEventListener("submit", function (event) {

    event.preventDefault();

    // Temporary frontend-only login
    // Backend authentication will be connected later.

    window.location.href = "onboarding.html";

});