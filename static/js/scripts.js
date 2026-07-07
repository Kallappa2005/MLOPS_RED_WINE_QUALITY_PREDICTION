document.addEventListener("DOMContentLoaded", function () {
    var revealNodes = document.querySelectorAll(".reveal");
    revealNodes.forEach(function (node, index) {
        node.style.animationDelay = (index * 100) + "ms";
        node.classList.add("is-visible");
    });

    var sampleButton = document.getElementById("sampleBtn");
    if (sampleButton) {
        sampleButton.addEventListener("click", function () {
            var sampleValues = {
                fixed_acidity: 7.4,
                volatile_acidity: 0.7,
                citric_acid: 0.0,
                residual_sugar: 1.9,
                chlorides: 0.076,
                free_sulfur_dioxide: 11.0,
                total_sulfur_dioxide: 34.0,
                density: 0.9978,
                pH: 3.51,
                sulphates: 0.56,
                alcohol: 9.4
            };

            Object.keys(sampleValues).forEach(function (name) {
                var input = document.querySelector("input[name='" + name + "']");
                if (input) {
                    input.value = sampleValues[name];
                }
            });
        });
    }

    // --- Mobile Navigation ---
    var hamburger = document.querySelector(".hamburger");
    var navMenu = document.querySelector(".menu");
    var overlay = document.querySelector(".nav-overlay");

    if (hamburger && navMenu) {
        function openNav() {
            hamburger.setAttribute("aria-expanded", "true");
            navMenu.classList.add("is-open");
            if (overlay) overlay.classList.add("is-visible");
            document.body.style.overflow = "hidden";
            // Focus first link inside the menu
            var firstLink = navMenu.querySelector("a");
            if (firstLink) firstLink.focus();
        }

        function closeNav() {
            hamburger.setAttribute("aria-expanded", "false");
            navMenu.classList.remove("is-open");
            if (overlay) overlay.classList.remove("is-visible");
            document.body.style.overflow = "";
            hamburger.focus();
        }

        function toggleNav() {
            var isOpen = hamburger.getAttribute("aria-expanded") === "true";
            if (isOpen) { closeNav(); } else { openNav(); }
        }

        hamburger.addEventListener("click", toggleNav);

        // Close on overlay click
        if (overlay) {
            overlay.addEventListener("click", closeNav);
        }

        // Close on Escape
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape") {
                closeNav();
            }
        });

        // Close menu when resizing to desktop width
        var resizeTimer;
        window.addEventListener("resize", function () {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(function () {
                if (window.innerWidth > 900) {
                    closeNav();
                }
            }, 100);
        });
    }
});