const TOPICS = [
  { name: "삼위일체론", desc: "성부·성자·성령의 관계" },
  { name: "기독론", desc: "그리스도의 인격과 사역" },
  { name: "종말론", desc: "기독교적 희망과 미래" },
  { name: "구원론", desc: "은혜·칭의·성화" },
  { name: "성령론", desc: "성령의 사역과 임재" },
  { name: "교회론", desc: "교회의 본질과 사명" },
  { name: "창조론", desc: "창조·섭리·생태" },
];

function renderTopics() {
  const grid = document.getElementById("topicCards");
  if (!grid) return;
  grid.innerHTML = TOPICS.map(t => `
    <a href="/symposium?topic=${encodeURIComponent(t.name)}" class="topic-card">
      <div class="topic-name">${t.name}</div>
      <div class="topic-desc">${t.desc}</div>
    </a>
  `).join("");
}

const ERA_GROUPS = [
  { label: "교부 시대",           color: "#c084fc", min: 0,    max: 499  },
  { label: "중세 초기",           color: "#a78bfa", min: 500,  max: 999  },
  { label: "중세",              color: "#818cf8", min: 1000, max: 1479 },
  { label: "종교개혁",            color: "#f97316", min: 1480, max: 1599 },
  { label: "경건주의 / 부흥운동",   color: "#fb923c", min: 1600, max: 1799 },
  { label: "근대 (19세기)",       color: "#38bdf8", min: 1800, max: 1899 },
  { label: "현대",              color: "#6b8afd", min: 1900, max: 9999 },
];

function getEra(born) {
  return ERA_GROUPS.find(e => born >= e.min && born <= e.max) || ERA_GROUPS[ERA_GROUPS.length - 1];
}

async function renderTheologians() {
  const grid = document.getElementById("theologianGrid");
  if (!grid) return;
  try {
    const res = await fetch("/api/authors");
    const authors = await res.json();

    const groups = new Map();
    for (const a of authors) {
      const era = getEra(a.born);
      if (!groups.has(era.label)) groups.set(era.label, { era, authors: [] });
      groups.get(era.label).authors.push(a);
    }

    let html = "";
    for (const { era, authors: eraAuthors } of groups.values()) {
      html += `<div class="landing-era-group">
        <div class="landing-era-header">
          <span class="era-group-dot" style="background:${era.color}"></span>
          <span class="landing-era-label">${era.label}</span>
        </div>
        <div class="theologian-grid">`;
      for (const a of eraAuthors) {
        html += `
          <a href="/symposium?invite=${encodeURIComponent(a.key)}" class="theologian-card theologian-card-link">
            <div class="theologian-name">${a.name_ko}</div>
            <div class="theologian-works">${a.tradition || ''}</div>
          </a>`;
      }
      html += `</div></div>`;
    }
    grid.innerHTML = html;
  } catch (e) {
    grid.innerHTML = '<p class="dim">신학자 목록을 불러올 수 없습니다.</p>';
  }
}

async function renderConfessions() {
  const grid = document.getElementById("confessionCards");
  if (!grid) return;
  try {
    const res = await fetch("/api/confessions");
    const confessions = await res.json();
    grid.innerHTML = confessions.map(c => {
      const shortTitle = c.title.split(" — ")[0];
      return `
      <a href="/symposium?confession=${encodeURIComponent(c.file)}&confession_name=${encodeURIComponent(shortTitle)}" class="topic-card">
        <div class="topic-name">${shortTitle}</div>
        <div class="topic-desc">${c.tradition} · ${c.year}</div>
      </a>`;
    }).join("");
  } catch (e) {
    grid.innerHTML = '<p class="dim">불러올 수 없습니다.</p>';
  }
}

renderTopics();
renderConfessions();
renderTheologians();
