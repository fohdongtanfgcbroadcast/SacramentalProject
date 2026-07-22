// 라이트/다크 테마 토글. <head>에서 즉시 실행되어 data-theme 를 먼저 적용(FOUC 방지).
// 저장값 우선, 없으면 prefers-color-scheme. 인라인 스크립트 없음(CSP script-src 'self' 준수).
(function () {
  var KEY = "symposium-theme";
  var root = document.documentElement;

  function apply(theme) {
    if (theme === "light") root.setAttribute("data-theme", "light");
    else root.removeAttribute("data-theme");
  }

  function initial() {
    var saved = null;
    try { saved = localStorage.getItem(KEY); } catch (e) {}
    if (saved === "light" || saved === "dark") return saved;
    if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) return "light";
    return "dark";
  }

  var theme = initial();
  apply(theme);

  function setupButton() {
    var btn = document.getElementById("themeToggle");
    if (!btn) return;
    function label() {
      btn.textContent = theme === "light" ? "☽" : "☀";  // 🌙 / ☀
      btn.title = theme === "light" ? "다크 모드로 전환" : "라이트 모드로 전환";
    }
    label();
    btn.addEventListener("click", function () {
      theme = theme === "light" ? "dark" : "light";
      try { localStorage.setItem(KEY, theme); } catch (e) {}
      apply(theme);
      label();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupButton);
  } else {
    setupButton();
  }
})();
