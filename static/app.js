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

async function renderTheologians() {
  const grid = document.getElementById("theologianGrid");
  if (!grid) return;
  try {
    const res = await fetch("/api/authors");
    const authors = await res.json();
    grid.innerHTML = authors.map(a => `
      <div class="theologian-card">
        <div class="theologian-name">${a.name_ko}</div>
        <div class="theologian-works">${a.work_count}권</div>
      </div>
    `).join("");
  } catch (e) {
    grid.innerHTML = '<p class="dim">신학자 목록을 불러올 수 없습니다.</p>';
  }
}

renderTopics();
renderTheologians();
