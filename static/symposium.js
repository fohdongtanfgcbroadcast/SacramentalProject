// --- State ---
let sessionId = null;
let selectedTheologians = [];
let allAuthors = [];
let isWaiting = false;

// --- DOM ---
const panel = document.getElementById("panel");
const theologianList = document.getElementById("theologianList");
const selectedCount = document.getElementById("selectedCount");
const btnStart = document.getElementById("btnStart");
const chatMessages = document.getElementById("chatMessages");
const recommendations = document.getElementById("recommendations");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const targetSelect = document.getElementById("targetSelect");

// --- URL params ---
const urlParams = new URLSearchParams(window.location.search);
const topicParam = urlParams.get("topic") || "";

// --- Init ---
async function init() {
  try {
    const res = await fetch("/api/authors");
    allAuthors = await res.json();
    renderTheologianList();
  } catch (e) {
    theologianList.innerHTML = '<p class="dim">불러오기 실패</p>';
  }
}

function renderTheologianList() {
  theologianList.innerHTML = allAuthors.map(a => `
    <label class="theologian-check-item" data-key="${a.key}">
      <input type="checkbox" value="${a.key}">
      <span class="theologian-check-name">${a.name_ko}</span>
      <span class="theologian-check-works">${a.work_count}권</span>
    </label>
  `).join("");

  theologianList.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.addEventListener("change", onSelectionChange);
  });
}

function onSelectionChange() {
  const checked = theologianList.querySelectorAll('input:checked');
  selectedTheologians = Array.from(checked).map(cb => cb.value);

  // 5명 초과 방지
  if (selectedTheologians.length > 5) {
    this.checked = false;
    selectedTheologians = selectedTheologians.filter(k => k !== this.value);
  }

  selectedCount.textContent = `(${selectedTheologians.length}/5)`;
  btnStart.disabled = selectedTheologians.length === 0;
}

// --- Start session ---
btnStart.addEventListener("click", async () => {
  if (selectedTheologians.length === 0) return;

  btnStart.disabled = true;
  btnStart.textContent = "시작 중...";

  try {
    const res = await fetch("/api/symposium/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ theologians: selectedTheologians, topic: topicParam }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    sessionId = data.session_id;

    // 체크박스 비활성화
    theologianList.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.disabled = true; });
    btnStart.classList.add("hidden");

    // 대상 셀렉트 채우기
    targetSelect.innerHTML = '<option value="">전체 라운드</option>' +
      data.theologians.map(t => `<option value="${t.key}">${t.name_ko}</option>`).join("");

    // 추천 질문 표시
    if (data.recommended_questions.length > 0) {
      recommendations.innerHTML =
        '<div class="rec-label">추천 질문</div>' +
        data.recommended_questions.map(q =>
          `<button type="button" class="rec-btn" data-q="${escapeAttr(q)}">${q}</button>`
        ).join("");
      recommendations.classList.remove("hidden");
      recommendations.querySelectorAll(".rec-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          chatInput.value = btn.dataset.q;
          recommendations.classList.add("hidden");
          chatInput.focus();
        });
      });
    }

    // 채팅 영역 초기화
    chatMessages.innerHTML = "";
    addSystemMessage("향연이 시작되었습니다. 질문을 입력하세요.");
    chatForm.classList.remove("hidden");
    chatInput.focus();
  } catch (e) {
    addSystemMessage("오류: " + e.message);
    btnStart.disabled = false;
    btnStart.textContent = "향연 시작";
  }
});

// --- Send message ---
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message || !sessionId || isWaiting) return;

  const target = targetSelect.value;
  chatInput.value = "";
  recommendations.classList.add("hidden");
  addUserMessage(message);

  if (target) {
    await sendDirect(message, target);
  } else {
    await sendRound(message);
  }
});

// --- Round (SSE) ---
async function sendRound(message) {
  isWaiting = true;
  setInputEnabled(false);

  // 첫 번째 신학자 대기 표시
  addWaitingMessage(selectedTheologians[0]);

  try {
    const res = await fetch("/api/symposium/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
    });

    if (!res.ok) throw new Error(await res.text());

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let theologianIndex = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith("data:")) {
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;
          try {
            const data = JSON.parse(jsonStr);
            if (data.done) continue;

            // 대기 메시지 제거, 답변 표시
            removeWaiting();
            addTheologianMessage(data);

            // 다음 신학자 대기 표시
            theologianIndex++;
            if (theologianIndex < selectedTheologians.length) {
              addWaitingMessage(selectedTheologians[theologianIndex]);
            }
          } catch (parseErr) { /* skip */ }
        }
      }
    }
  } catch (e) {
    removeWaiting();
    addSystemMessage("오류: " + e.message);
  }

  isWaiting = false;
  setInputEnabled(true);
  chatInput.focus();
}

// --- Direct ---
async function sendDirect(message, target) {
  isWaiting = true;
  setInputEnabled(false);
  addWaitingMessage(target);

  try {
    const res = await fetch("/api/symposium/direct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message, target }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    removeWaiting();
    addTheologianMessage(data);
  } catch (e) {
    removeWaiting();
    addSystemMessage("오류: " + e.message);
  }

  isWaiting = false;
  setInputEnabled(true);
  chatInput.focus();
}

// --- DOM Helpers ---
function addUserMessage(text) {
  const div = document.createElement("div");
  div.className = "chat-msg chat-msg-user";
  div.textContent = text;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function addTheologianMessage(data) {
  const div = document.createElement("div");
  div.className = "chat-msg chat-msg-theologian";

  const header = document.createElement("div");
  header.className = "chat-msg-header";
  header.innerHTML = `<span class="speaker-name">${escapeHtml(data.name_ko)}</span>` +
    (data.tradition ? `<span class="speaker-era">${escapeHtml(data.tradition)}</span>` : "");

  const body = document.createElement("div");
  body.className = "chat-msg-body";
  body.textContent = data.text;

  div.appendChild(header);
  div.appendChild(body);

  // 출처 (접기)
  if (data.sources && data.sources.length > 0) {
    const details = document.createElement("details");
    details.className = "chat-sources";
    details.innerHTML = `<summary>참고 자료 (${data.sources.length})</summary>` +
      data.sources.map(s =>
        `<div class="source-item">[${escapeHtml(s.title)}, p.${s.page}] ${escapeHtml(s.text)}</div>`
      ).join("");
    div.appendChild(details);
  }

  chatMessages.appendChild(div);
  scrollToBottom();
}

function addWaitingMessage(authorKey) {
  const author = allAuthors.find(a => a.key === authorKey);
  const name = author ? author.name_ko : authorKey;
  const div = document.createElement("div");
  div.className = "chat-msg chat-msg-waiting";
  div.id = "waiting-indicator";
  div.innerHTML = `<span class="waiting-dots"></span> ${escapeHtml(name)}가 생각하고 있습니다...`;
  chatMessages.appendChild(div);
  scrollToBottom();
  return div;
}

function removeWaiting() {
  const el = document.getElementById("waiting-indicator");
  if (el) el.remove();
}

function addSystemMessage(text) {
  const div = document.createElement("div");
  div.className = "chat-msg chat-msg-system";
  div.textContent = text;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function setInputEnabled(enabled) {
  chatInput.disabled = !enabled;
  chatForm.querySelector("button").disabled = !enabled;
  targetSelect.disabled = !enabled;
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function escapeAttr(str) {
  return str.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// --- Go ---
init();
