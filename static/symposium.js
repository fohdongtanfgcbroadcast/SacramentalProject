// --- State ---
let sessionId = null;
let selectedTheologians = [];
let sessionTheologians = []; // {key, name_ko, ...} from /api/symposium/start
let allAuthors = [];
let isWaiting = false;
let currentQuestion = "";
let spokenInRound = new Set(); // 이번 라운드에서 발언한 신학자

// --- DOM ---
const theologianList = document.getElementById("theologianList");
const selectedCount = document.getElementById("selectedCount");
const btnStart = document.getElementById("btnStart");
const chatMessages = document.getElementById("chatMessages");
const recommendations = document.getElementById("recommendations");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const speakerPicker = document.getElementById("speakerPicker");
const pickerLabel = document.getElementById("pickerLabel");
const pickerButtons = document.getElementById("pickerButtons");

// --- URL params ---
const urlParams = new URLSearchParams(window.location.search);
const topicParam = urlParams.get("topic") || "";
const inviteParams = urlParams.getAll("invite");
const confessionParam = urlParams.get("confession") || "";
const confessionNameParam = urlParams.get("confession_name") || "";

let confessionsList = []; // 신앙고백서 개별 문서

// --- Init ---
async function init() {
  try {
    const res = await fetch("/api/authors");
    allAuthors = await res.json();
    try {
      const confRes = await fetch("/api/confessions");
      confessionsList = await confRes.json();
    } catch (e) {
      confessionsList = [];
    }
    renderTheologianList();
    // 신앙고백서 토론 모드 — 전문 로드
    if (confessionParam) {
      const ctx = document.getElementById("confessionContext");
      ctx.innerHTML = `<span class="confession-context-label">토론 대상</span> <strong>${escapeHtml(confessionNameParam || confessionParam)}</strong>`;
      ctx.classList.remove("hidden");
      loadConfessionText(confessionParam, confessionNameParam);
    }
    // URL에서 invite 파라미터로 미리 선택
    if (inviteParams.length > 0) {
      inviteParams.forEach(key => {
        const cb = theologianList.querySelector(`input[value="${key}"]`);
        if (cb && !cb.checked) {
          cb.checked = true;
          cb.dispatchEvent(new Event("change"));
        }
      });
    }
  } catch (e) {
    theologianList.innerHTML = '<p class="dim">불러오기 실패</p>';
  }
}

const ERA_GROUPS = [
  { label: "교부 시대",           color: "#c084fc", min: 0,    max: 499,  shape: "circle" },
  { label: "중세 초기",           color: "#a78bfa", min: 500,  max: 999,  shape: "square" },
  { label: "중세",              color: "#818cf8", min: 1000, max: 1479, shape: "diamond" },
  { label: "종교개혁",            color: "#f97316", min: 1480, max: 1599, shape: "triangle" },
  { label: "경건주의 / 부흥운동",   color: "#fb923c", min: 1600, max: 1799, shape: "hexagon" },
  { label: "근대 (19세기)",       color: "#38bdf8", min: 1800, max: 1899, shape: "star" },
  { label: "현대",              color: "#6b8afd", min: 1900, max: 9999, shape: "cross" },
];

function getEra(born) {
  return ERA_GROUPS.find(e => born >= e.min && born <= e.max) || ERA_GROUPS[ERA_GROUPS.length - 1];
}

function renderTheologianList() {
  // 신학자 시대별 그룹핑 (confessions 제외)
  const groups = new Map();
  for (const a of allAuthors) {
    if (a.key === "confessions") continue;
    const era = getEra(a.born);
    if (!groups.has(era.label)) groups.set(era.label, { era, authors: [], confessions: [] });
    groups.get(era.label).authors.push(a);
  }

  // 신앙고백서를 시대별로 분배
  for (const c of confessionsList) {
    const era = getEra(c.year);
    if (!groups.has(era.label)) groups.set(era.label, { era, authors: [], confessions: [] });
    groups.get(era.label).confessions.push(c);
  }

  let html = "";
  for (const { era, authors, confessions } of groups.values()) {
    html += `<div class="era-group">
      <div class="era-group-header">
        <span class="era-group-dot" style="background:${era.color}"></span>
        <span class="era-group-label">${era.label}</span>
      </div>`;
    for (const a of authors) {
      html += `
      <label class="theologian-check-item" data-key="${escapeAttr(a.key)}">
        <input type="checkbox" value="${escapeAttr(a.key)}">
        <span class="theologian-check-avatar" style="background:${era.color}18;color:${era.color}">
          <span class="avatar-shape shape-${era.shape}"></span>
        </span>
        <span class="theologian-check-name">${escapeHtml(a.name_ko)}</span>
        <span class="theologian-check-works">${escapeHtml(a.work_count)}권</span>
      </label>`;
    }
    // 해당 시대 신앙고백서 (교부 시대 제외)
    if (confessions.length > 0 && era.label !== "교부 시대") {
      html += `<div class="confession-sub-label">신앙고백서</div>`;
      for (const c of confessions) {
        const shortTitle = c.title.split(" — ")[0];
        html += `
        <div class="confession-item">
          <span class="confession-name">${escapeHtml(shortTitle)}</span>
          <span class="confession-year">${escapeHtml(c.year)}</span>
        </div>`;
      }
    }
    html += `</div>`;
  }

  theologianList.innerHTML = html;

  theologianList.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.addEventListener("change", onSelectionChange);
  });
}

function onSelectionChange() {
  const checked = theologianList.querySelectorAll('input:checked');
  selectedTheologians = Array.from(checked).map(cb => cb.value);

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
      body: JSON.stringify({ theologians: selectedTheologians, topic: topicParam, confession: confessionParam, confession_name: confessionNameParam }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    sessionId = data.session_id;
    sessionTheologians = data.theologians;

    // 체크박스 비활성화
    theologianList.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.disabled = true; });
    btnStart.classList.add("hidden");

    // 추천 질문 표시
    if (data.recommended_questions.length > 0) {
      recommendations.innerHTML =
        '<div class="rec-label">추천 질문</div>' +
        data.recommended_questions.map(q =>
          `<button type="button" class="rec-btn" data-q="${escapeAttr(q)}">${escapeHtml(q)}</button>`
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
    if (data.confession_name) {
      addSystemMessage(`"${data.confession_name}" 토론이 시작되었습니다. 이 고백서에 대해 질문하고 발언자를 지명하세요.`);
    } else {
      addSystemMessage("향연이 시작되었습니다. 좌장으로서 질문을 입력하고, 발언자를 지명하세요.");
    }
    chatForm.classList.remove("hidden");
    chatInput.focus();
  } catch (e) {
    addSystemMessage("오류: " + e.message);
    btnStart.disabled = false;
    btnStart.textContent = "향연 시작";
  }
});

// --- Send message (질문 입력) ---
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message || !sessionId || isWaiting) return;

  chatInput.value = "";
  recommendations.classList.add("hidden");
  currentQuestion = message;
  spokenInRound = new Set();

  addUserMessage(message);
  showSpeakerPicker();
});

// --- Speaker Picker (좌장 모드) ---
function showSpeakerPicker() {
  const remaining = sessionTheologians.filter(t => !spokenInRound.has(t.key));

  if (remaining.length === 0) {
    // 모든 신학자가 발언 완료
    hideSpeakerPicker();
    addSystemMessage("이번 라운드가 완료되었습니다. 새 질문을 입력하세요.");
    setInputEnabled(true);
    chatInput.focus();
    return;
  }

  const isFirst = spokenInRound.size === 0;
  pickerLabel.textContent = isFirst ? "누가 먼저 답할까요?" : "다음 발언자를 선택하세요";

  pickerButtons.innerHTML = remaining.map(t => {
    const tAuthor = allAuthors.find(a => a.key === t.key);
    const tEra = tAuthor ? getEra(tAuthor.born) : ERA_GROUPS[0];
    return `
    <button type="button" class="picker-btn" data-key="${t.key}" data-name="${escapeAttr(t.name_ko)}">
      <span class="picker-avatar" style="background:${tEra.color}18;color:${tEra.color}">
        <span class="avatar-shape shape-${tEra.shape}"></span>
      </span>
      <span>${escapeHtml(t.name_ko)}</span>
    </button>`;
  }).join("") + (spokenInRound.size > 0 ? `
    <button type="button" class="picker-btn picker-btn-skip">라운드 종료</button>
  ` : "");

  pickerButtons.querySelectorAll(".picker-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      if (btn.classList.contains("picker-btn-skip")) {
        hideSpeakerPicker();
        addSystemMessage("라운드를 종료합니다. 새 질문을 입력하세요.");
        setInputEnabled(true);
        chatInput.focus();
        return;
      }
      const key = btn.dataset.key;
      hideSpeakerPicker();
      callTheologian(key);
    });
  });

  speakerPicker.classList.remove("hidden");
  setInputEnabled(false);
  scrollToBottom();
}

function hideSpeakerPicker() {
  speakerPicker.classList.add("hidden");
}

// --- Call theologian (direct) ---
async function callTheologian(authorKey) {
  isWaiting = true;
  addWaitingMessage(authorKey);

  try {
    const res = await fetch("/api/symposium/direct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: currentQuestion, target: authorKey }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    removeWaiting();
    addTheologianMessage(data);
    spokenInRound.add(authorKey);
  } catch (e) {
    removeWaiting();
    addSystemMessage("오류: " + e.message);
  }

  isWaiting = false;
  showSpeakerPicker();
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
  const t = sessionTheologians.find(a => a.key === authorKey);
  const name = t ? t.name_ko : authorKey;
  const div = document.createElement("div");
  div.className = "chat-msg chat-msg-waiting";
  div.id = "waiting-indicator";
  div.innerHTML = `<span class="waiting-dots"></span> ${escapeHtml(name)}가 생각하고 있습니다...`;
  chatMessages.appendChild(div);
  scrollToBottom();
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
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function loadConfessionText(filename, name) {
  const welcome = chatMessages.querySelector(".chat-welcome");
  if (welcome) welcome.innerHTML = `<p>${escapeHtml(name)}을(를) 불러오는 중...</p>`;
  try {
    const res = await fetch(`/api/confession-text/${encodeURIComponent(filename)}`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    chatMessages.innerHTML = "";
    const div = document.createElement("div");
    div.className = "confession-fulltext";

    if (data.mode === "toc") {
      div.innerHTML = `<div class="confession-fulltext-title">${escapeHtml(name)} — 목차</div>
        <div class="confession-fulltext-body confession-toc">${escapeHtml(data.text)}</div>
        <div class="confession-truncated">전문이 길어 목차만 표시합니다. 특정 조항에 대해 질문하시면 RAG 검색으로 해당 부분을 자동 추출합니다.</div>`;
    } else {
      const langNote = data.lang === "en" ? '<div class="confession-truncated">영문 원본입니다. 신학자들은 한국어로 토론합니다.</div>' : '';
      div.innerHTML = `<div class="confession-fulltext-title">${escapeHtml(name)}</div>
        <div class="confession-fulltext-body">${escapeHtml(data.text)}</div>
        ${langNote}`;
    }

    chatMessages.appendChild(div);
    scrollToBottom();
  } catch (e) {
    if (welcome) welcome.innerHTML = `<p>전문을 불러올 수 없습니다.</p>`;
  }
}

function escapeAttr(str) {
  return str.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// --- Go ---
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
