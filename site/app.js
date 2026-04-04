const input = document.getElementById("word-input");
const button = document.getElementById("word-button");
const result = document.getElementById("word-result");
const suggestions = document.getElementById("word-suggestions");

let scoresPromise = null;
let scoresData = null;
let topSuggestedWords = [];
let healthChipWordsPromise = null;

const MAX_SUGGESTION_CHIPS = 45;

const HEALTH_SUGGESTION_SEEDS = [
  "brain",
  "heart",
  "muscle",
  "eye",
  "lung",
  "liver",
  "kidney",
  "blood",
  "cancer",
  "tumor",
  "stroke",
  "diabetes",
  "obesity",
  "pain",
  "immune",
  "infection",
  "vaccine",
  "therapy",
  "neuro",
  "cardiac",
  "oncology",
  "retina",
  "vascular",
  "arthritis",
  "cardiovascular",
  "pulmonary",
  "respiratory",
  "renal",
  "bone",
  "joint",
  "breast",
  "prostate",
  "lipid",
  "metabolic",
  "arterial",
  "gastrointestinal",
  "neurological",
  "cholesterol",
  "mental",
  "depression",
  "anxiety",
  "sleep",
  "pediatric",
  "pregnancy",
  "diabetic",
  "infections",
  "infectious",
  "electrocardiogram"
];

const HEALTH_WORD_PATTERNS = [
  /^brain$/,
  /^heart$/,
  /^muscle/,
  /^eye$/,
  /^lung/,
  /^liver$/,
  /^kidney$/,
  /^renal/,
  /^blood$/,
  /^cancer/,
  /^tumor/,
  /^stroke$/,
  /^diabet/,
  /^obes/,
  /^pain$/,
  /^immune/,
  /^infect/,
  /^vaccine/,
  /^therapy/,
  /^neuro/,
  /^card/,
  /^oncolog/,
  /^retina/,
  /^vascular/,
  /^arthritis$/,
  /^respirat/,
  /^pulmon/,
  /^bone$/,
  /^joint/,
  /^cardiovascular/,
  /^breast$/,
  /^prostate$/,
  /^lipid/,
  /^metabolic/,
  /^arterial/,
  /^gastrointestinal/,
  /^neurolog/,
  /^cholesterol/,
  /^pregnancy$/,
  /^mental$/,
  /^depress/,
  /^anxiety$/,
  /^sleep$/,
  /^pediatric$/,
  /^electrocardiogram$/
];

const HEALTH_WORD_EXCLUSIONS = new Set([
  "masking",
  "experimental",
  "delivery",
  "concomitant"
]);

function isValidSuggestionWord(word) {
  return (
    typeof word === "string" &&
    /^[a-z0-9]+$/.test(word) &&
    word.length > 1 &&
    word !== "nan" &&
    word !== "null" &&
    word !== "none"
  );
}

function isHealthThemeWord(word) {
  if (!isValidSuggestionWord(word) || HEALTH_WORD_EXCLUSIONS.has(word)) {
    return false;
  }
  return HEALTH_WORD_PATTERNS.some((pattern) => pattern.test(word));
}

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

function loadHealthChipWords() {
  if (healthChipWordsPromise) {
    return healthChipWordsPromise;
  }

  healthChipWordsPromise = fetch("chip_words_health_filtered.txt")
    .then((response) => {
      if (!response.ok) {
        throw new Error("Missing chip_words_health_filtered.txt");
      }
      return response.text();
    })
    .then((text) => {
      return text
        .split(/\r?\n/)
        .map((word) => word.trim().toLowerCase())
        .filter((word) => isValidSuggestionWord(word));
    })
    .catch(() => []);

  return healthChipWordsPromise;
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
      const entries = Object.entries(data.data || {});
      const rankedWords = entries
        .filter(([word, entry]) => isValidSuggestionWord(word) && (entry.count || 0) > 0)
        .sort((a, b) => (b[1].count || 0) - (a[1].count || 0))
        .map(([word]) => word);

      const healthWords = HEALTH_SUGGESTION_SEEDS.filter(
        (word) => data.data && data.data[word] && isValidSuggestionWord(word)
      );

      const healthWordsFromData = rankedWords
        .filter((word) => isHealthThemeWord(word))
        .slice(0, 80);

      const fallbackWords = [
        ...new Set([...healthWords, ...healthWordsFromData, ...rankedWords])
      ].slice(0, 80);

      return loadHealthChipWords().then((fileWords) => {
        const fileBackedWords = fileWords.filter((word) => data.data && data.data[word]);
        topSuggestedWords = fileBackedWords.length > 0 ? fileBackedWords : fallbackWords;
        return data;
      });
    })
    .catch(() => {
      renderMessage(
        "word_scores.json is missing. Run scripts/build_word_scores.py to generate it."
      );
      throw new Error("Missing word_scores.json");
    });
  return scoresPromise;
}

function getSuggestionWords(rawValue) {
  if (!scoresData || !scoresData.data) {
    return [];
  }

  const defaultSuggestions = topSuggestedWords.slice(0, MAX_SUGGESTION_CHIPS);

  const trimmed = rawValue.trim().toLowerCase();
  if (!trimmed) {
    return defaultSuggestions;
  }

  if (!/^[a-z0-9]+$/.test(trimmed)) {
    return defaultSuggestions;
  }

  if (scoresData.data && scoresData.data[trimmed]) {
    return defaultSuggestions
      .filter((word) => word !== trimmed)
      .slice(0, MAX_SUGGESTION_CHIPS);
  }

  const prefixMatches = [];
  for (const word of topSuggestedWords) {
    if (word.startsWith(trimmed) && word !== trimmed) {
      prefixMatches.push(word);
    }
    if (prefixMatches.length >= MAX_SUGGESTION_CHIPS) {
      break;
    }
  }

  if (prefixMatches.length >= MAX_SUGGESTION_CHIPS) {
    return prefixMatches;
  }

  if (prefixMatches.length > 0) {
    return prefixMatches;
  }

  return defaultSuggestions
    .filter((word) => word !== trimmed)
    .slice(0, MAX_SUGGESTION_CHIPS);
}

function renderSuggestions() {
  if (!suggestions) {
    return;
  }

  suggestions.innerHTML = "";
  const words = getSuggestionWords(input.value);
  if (words.length === 0) {
    return;
  }

  const fragment = document.createDocumentFragment();
  words.forEach((word) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "suggestion-chip";
    chip.textContent = word;
    chip.addEventListener("click", () => {
      input.value = word;
      analyzeWord();
      renderSuggestions();
      input.focus();
    });
    fragment.appendChild(chip);
  });

  suggestions.appendChild(fragment);
}

function renderStats(word, entry, meta) {
  const baseline = meta && typeof meta.global_success_rate === "number"
    ? formatPercent(meta.global_success_rate)
    : "n/a";
  const assoc = entry.assoc_score.toFixed(3);
  const baselineRate = meta && typeof meta.global_success_rate === "number"
    ? formatPercent(meta.global_success_rate)
    : "the overall average";

  const metricCards = [
    {
      label: "Trials containing word",
      value: entry.count,
      iconClass: "icon-trials",
      iconText: "#",
      tooltip: "How many study records mention this word. Bigger number means this word shows up more often in the dataset."
    },
    {
      label: "Success rate",
      value: formatPercent(entry.success_rate),
      iconClass: "icon-success",
      iconText: "%",
      tooltip: "Out of trials that mention this word, this is the share marked successful."
    },
    {
      label: "Lift vs baseline",
      value: formatPoints(entry.lift),
      iconClass: "icon-lift",
      iconText: "+",
      tooltip: `How much this word's success rate is above or below the typical rate (${baselineRate}). Positive is better than average; negative is lower than average.`
    },
    {
      label: "Embedding association",
      value: assoc,
      iconClass: "icon-association",
      iconText: "~",
      tooltip: "A pattern-match score: positive means this word appears in studies that look more like successful ones; negative means it looks more like unsuccessful ones."
    }
  ];

  const metricMarkup = metricCards
    .map((metric) => `
      <div class="metric">
        <div class="metric-label-row">
          <span class="label">${metric.label}</span>
          <span class="metric-tooltip-wrap" tabindex="0" aria-label="About ${metric.label}">
            <span class="metric-info-icon ${metric.iconClass}" aria-hidden="true">${metric.iconText}</span>
            <span class="metric-tooltip" role="tooltip">${metric.tooltip}</span>
          </span>
        </div>
        <span class="value">${metric.value}</span>
      </div>
    `)
    .join("");

  result.innerHTML = `
    <div>
      <div class="result-title-row">
        <strong>${word}</strong>
      </div>
      <span class="message">Based on ${entry.count} trials. Baseline success: ${baseline}.</span>
    </div>
    <div class="metrics">
      ${metricMarkup}
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
input.addEventListener("input", renderSuggestions);

renderMessage("Type a word to see results.");
loadScores().then(renderSuggestions).catch(() => {});

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
