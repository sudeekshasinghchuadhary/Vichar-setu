// ===============================
// GET HTML ELEMENTS
// ===============================

const loginPage = document.getElementById("loginPage");
const createAccountPage = document.getElementById("createAccountPage");

const showCreateAccount = document.getElementById("showCreateAccount");
const showLogin = document.getElementById("showLogin");

const loginForm = document.getElementById("loginForm");
const createAccountForm = document.getElementById("createAccountForm");

const loginMessage = document.getElementById("loginMessage");
const createMessage = document.getElementById("createMessage");


// ===============================
// OPEN CREATE ACCOUNT PAGE
// ===============================

showCreateAccount.addEventListener("click", function () {

    // Hide login page
    loginPage.classList.add("hidden");

    // Show create account page
    createAccountPage.classList.remove("hidden");

});


// ===============================
// BACK TO LOGIN PAGE
// ===============================

showLogin.addEventListener("click", function () {

    // Hide create account page
    createAccountPage.classList.add("hidden");

    // Show login page
    loginPage.classList.remove("hidden");

});


// ===============================
// CREATE ACCOUNT
// ===============================

createAccountForm.addEventListener("submit", function (event) {

    // Stop page from refreshing
    event.preventDefault();


    // Get values from inputs
    const name = document.getElementById("name").value;

    const email = document.getElementById("createEmail").value;

    const password =
        document.getElementById("createPassword").value;

    const confirmPassword =
        document.getElementById("confirmPassword").value;


    // ===============================
    // CHECK PASSWORD
    // ===============================

    if (password !== confirmPassword) {

        createMessage.textContent = "Passwords do not match!";

        createMessage.className = "error";

        return;
    }


    // ===============================
    // CREATE USER OBJECT
    // ===============================

    const user = {
        name: name,
        email: email,
        password: password
    };


    // ===============================
    // SAVE ACCOUNT
    // ===============================

    localStorage.setItem("user", JSON.stringify(user));


    // Show success message
    createMessage.textContent =
        "Account created successfully!";

    createMessage.className = "success";


    // Clear create account form
    createAccountForm.reset();


    // ===============================
    // GO BACK TO LOGIN
    // ===============================

    setTimeout(function () {

        // Hide create account page
        createAccountPage.classList.add("hidden");

        // Show login page
        loginPage.classList.remove("hidden");

        // Show message on login page
        loginMessage.textContent =
            "Account created! Please login.";

        loginMessage.className = "success";

    }, 1000);

});


// ===============================
// LOGIN
// ===============================

loginForm.addEventListener("submit", function (event) {

    // Stop page from refreshing
    event.preventDefault();


    // Get login values
    const email =
        document.getElementById("loginEmail").value;

    const password =
        document.getElementById("loginPassword").value;


    // ===============================
    // GET SAVED USER
    // ===============================

    const savedUser = localStorage.getItem("user");


    // Check whether account exists
    if (savedUser === null) {

        loginMessage.textContent =
            "No account found. Please create an account.";

        loginMessage.className = "error";

        return;
    }


    // Convert saved JSON into JavaScript object
    const user = JSON.parse(savedUser);


    // ===============================
    // CHECK LOGIN DETAILS
    // ===============================

    if (email === user.email && password === user.password) {

        loginMessage.textContent =
            "Login successful! Welcome " + user.name;

        loginMessage.className = "success";

    } 
    
    else {

        loginMessage.textContent =
            "Invalid email or password!";

        loginMessage.className = "error";

    }

});