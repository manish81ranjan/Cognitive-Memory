(function () {
  const ready = setInterval(() => {
    const hamburger = document.getElementById("hamburger");
    const menu = document.getElementById("mobileMenu");
    const chatBox = document.getElementById("chatbot");
    const closeBtn = document.getElementById("closeMenuBtn"); // your new close button

    if (!hamburger || !menu) return;
    clearInterval(ready);

    // Initial menu state
    closeMenu();

    // Hamburger toggle
    hamburger.onclick = toggleMenu;

    // Close button click
    if (closeBtn) {
      closeBtn.onclick = closeMenu;
    }

    // Menu buttons click
    menu.querySelectorAll("a, button, [data-target]").forEach((btn) => {
      btn.onclick = () => {
        const page = btn.dataset.target;

        // Show page if data-target exists
        if (page) {
          document.querySelectorAll(".page").forEach((p) =>
            p.classList.add("hidden")
          );
          document.getElementById(page)?.classList.remove("hidden");
        }

        // Open chatbot if chat button clicked
        if (btn.id === "chatBtnMobile" && chatBox) {
          chatBox.classList.remove("hidden");
        }

        // Close menu after action
        closeMenu();
      };
    });

    // Close menu if clicking outside
    menu.onclick = (e) => {
      if (e.target === menu) closeMenu();
    };

    // ====================
    // Functions
    // ====================
    function toggleMenu() {
      menu.classList.contains("open") ? closeMenu() : openMenu();
    }

    function openMenu() {
      menu.classList.add("open");
      hamburger.innerHTML = "✕"; // optional: change hamburger to X
    }

    function closeMenu() {
      menu.classList.remove("open");
      hamburger.innerHTML = "☰"; // optional: restore hamburger
    }

    console.log("🔥 Mobile menu system armed with close button support");
  }, 200);
})();

{/* <script>
const h = document.getElementById("hamburger");
const m = document.getElementById("mobileMenu");

h.onclick = () => {
  h.classList.toggle("active");
  m.classList.toggle("open");
};

m.querySelectorAll("button,a").forEach(el=>{
  el.onclick = () => {
    h.classList.remove("active");
    m.classList.remove("open");
  };
});
</script> */}
