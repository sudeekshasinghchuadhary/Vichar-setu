/* =========================
   SAKSHAM AI — NEAR MISS
========================= */


/* =========================
   SMALL CARD INTERACTION
========================= */

const buttons = document.querySelectorAll(
    ".small-near-card button"
);


buttons.forEach((button) => {

    button.addEventListener("click", () => {

        alert(
            "This opportunity may become a stronger match when your profile changes. Saksham AI will re-check your eligibility."
        );

    });

});


/* =========================
   PROFILE UPDATE
========================= */

const profileButton =
    document.querySelector(".profile-cta a");


if (profileButton) {

    profileButton.addEventListener("click", () => {

        localStorage.setItem(
            "sakshamProfileUpdate",
            "true"
        );

    });

}


/* =========================
   SIMPLE ENTRY ANIMATION
========================= */

const nearCard =
    document.querySelector(".near-card");


const smallCards =
    document.querySelectorAll(".small-near-card");


window.addEventListener("load", () => {

    if (nearCard) {

        nearCard.style.opacity = "0";
        nearCard.style.transform = "translateY(20px)";

        setTimeout(() => {

            nearCard.style.transition =
                "opacity 0.6s ease, transform 0.6s ease";

            nearCard.style.opacity = "1";
            nearCard.style.transform = "translateY(0)";

        }, 100);

    }


    smallCards.forEach((card, index) => {

        card.style.opacity = "0";
        card.style.transform = "translateY(20px)";

        setTimeout(() => {

            card.style.transition =
                "opacity 0.5s ease, transform 0.5s ease";

            card.style.opacity = "1";
            card.style.transform = "translateY(0)";

        }, 250 + (index * 120));

    });

});