const input = document.getElementById("word-input");
const button = document.getElementById("word-button");
const result = document.getElementById("word-result");

let scoresPromise = null;
let scoresData = null;

function renderMessage(message) {
  result.innerHTML = `<p class="message">${message}</p>`;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatPoints(value) {
  const points = (value * 100).toFixed(1);
  return `${points} pp`;
}

function normalizeWord(value) {
  const tokens = value.toLowerCase().match(/[a-z0-9]+/g);
  if (!tokens || tokens.length === 0) {
    return null;
  }
  if (tokens.length > 1) {
    return "__MULTI__";
  }
  return tokens[0];
}

function loadScores() {
  if (scoresPromise) {
    return scoresPromise;
  }
  scoresPromise = fetch("word_scores.json")
    .then((response) => {
      if (!response.ok) {
        throw new Error("Missing word_scores.json");
      }
      return response.json();
    })
    .then((data) => {
      scoresData = data;
      return data;
    })
    .catch(() => {
      renderMessage(
        "word_scores.json is missing. Run scripts/build_word_scores.py to generate it."
      );
      throw new Error("Missing word_scores.json");
    });
  return scoresPromise;
}

function renderStats(word, entry, meta) {
  const baseline = meta && typeof meta.global_success_rate === "number"
    ? formatPercent(meta.global_success_rate)
    : "n/a";
  const assoc = entry.assoc_score.toFixed(3);

  result.innerHTML = `
    <div>
      <strong>${word}</strong>
      <span class="message">Based on ${entry.count} trials. Baseline success: ${baseline}.</span>
    </div>
    <div class="metrics">
      <div class="metric">
        <span class="label">Trials containing word</span>
        <span class="value">${entry.count}</span>
      </div>
      <div class="metric">
        <span class="label">Success rate</span>
        <span class="value">${formatPercent(entry.success_rate)}</span>
      </div>
      <div class="metric">
        <span class="label">Lift vs baseline</span>
        <span class="value">${formatPoints(entry.lift)}</span>
      </div>
      <div class="metric">
        <span class="label">Embedding association</span>
        <span class="value">${assoc}</span>
      </div>
    </div>
  `;
}

function analyzeWord() {
  const normalized = normalizeWord(input.value.trim());
  if (!normalized) {
    renderMessage("Enter a single word using letters or numbers.");
    return;
  }
  if (normalized === "__MULTI__") {
    renderMessage("Please enter a single word (no spaces).");
    return;
  }

  loadScores()
    .then((data) => {
      const entry = data.data ? data.data[normalized] : null;
      if (!entry) {
        renderMessage("No data for that word. Try a more common term.");
        return;
      }
      renderStats(normalized, entry, data.meta || {});
    })
    .catch(() => {});
}

button.addEventListener("click", analyzeWord);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    analyzeWord();
  }
});

renderMessage("Type a word to see results.");

const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");
const lightboxCaption = document.getElementById("lightbox-caption");
const lightboxClosers = document.querySelectorAll("[data-lightbox-close]");
const cards = document.querySelectorAll(".gallery .card");

function openLightbox(image, captionText) {
  lightboxImage.src = image.src;
  lightboxImage.alt = image.alt || "Expanded visualization";
  lightboxCaption.textContent = captionText;
  lightbox.classList.add("open");
  lightbox.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  lightbox.classList.remove("open");
  lightbox.setAttribute("aria-hidden", "true");
  lightboxImage.src = "";
  lightboxCaption.textContent = "";
  document.body.style.overflow = "";
}

cards.forEach((card) => {
  card.setAttribute("tabindex", "0");
  card.setAttribute("role", "button");

  const img = card.querySelector("img");
  const caption = card.querySelector("figcaption");
  const captionText = caption ? caption.textContent.trim() : "";

  card.addEventListener("click", () => {
    if (!img) {
      return;
    }
    openLightbox(img, captionText);
  });

  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (!img) {
        return;
      }
      openLightbox(img, captionText);
    }
  });
});

lightboxClosers.forEach((closer) => {
  closer.addEventListener("click", closeLightbox);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && lightbox.classList.contains("open")) {
    closeLightbox();
  }
});
