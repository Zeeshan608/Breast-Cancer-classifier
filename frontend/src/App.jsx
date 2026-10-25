import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

/* Address of the Python (FastAPI) server that holds the trained model.
   Set VITE_API_URL in a .env file (see .env.example). */
const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

/* ------------------------------------------------------------------ */
/* The 30 inputs, in exactly the order the model was trained on.       */
/* range = smallest and largest value seen in the training data.       */
/* ------------------------------------------------------------------ */
const MEASURES = [
  { key: "radius", label: "Radius", hint: "Distance from the center of the nucleus to its edge",
    range: { mean: [6.981, 28.11], se: [0.1115, 2.873], worst: [7.93, 36.04] } },
  { key: "texture", label: "Texture", hint: "Variation in gray shades across the nucleus",
    range: { mean: [9.71, 39.28], se: [0.3602, 4.885], worst: [12.02, 49.54] } },
  { key: "perimeter", label: "Perimeter", hint: "Length of the nucleus outline",
    range: { mean: [43.79, 188.5], se: [0.757, 21.98], worst: [50.41, 251.2] } },
  { key: "area", label: "Area", hint: "Size of the nucleus",
    range: { mean: [143.5, 2501], se: [6.802, 542.2], worst: [185.2, 4254] } },
  { key: "smoothness", label: "Smoothness", hint: "How much the radius changes along the edge",
    range: { mean: [0.0526, 0.1634], se: [0.0017, 0.0311], worst: [0.0712, 0.2226] } },
  { key: "compactness", label: "Compactness", hint: "How tightly packed the shape is (perimeter² ÷ area − 1)",
    range: { mean: [0.0194, 0.3454], se: [0.0023, 0.1354], worst: [0.0273, 1.058] } },
  { key: "concavity", label: "Concavity", hint: "How deep the dents in the outline are",
    range: { mean: [0, 0.4268], se: [0, 0.396], worst: [0, 1.252] } },
  { key: "concave points", label: "Concave points", hint: "How many dents the outline has",
    range: { mean: [0, 0.2012], se: [0, 0.0528], worst: [0, 0.291] } },
  { key: "symmetry", label: "Symmetry", hint: "How even the nucleus looks on both sides",
    range: { mean: [0.106, 0.304], se: [0.0079, 0.079], worst: [0.1565, 0.6638] } },
  { key: "fractal_dimension", label: "Fractal dimension", hint: "How rough or jagged the outline is",
    range: { mean: [0.05, 0.0974], se: [0.0009, 0.0298], worst: [0.055, 0.2075] } },
];

const GROUPS = [
  { id: "mean", title: "Mean", blurb: "The average over all cell nuclei in the image." },
  { id: "se", title: "Standard error", blurb: "How much each measurement varies from one nucleus to the next." },
  { id: "worst", title: "Worst", blurb: "The most extreme values found, averaged over the three largest." },
];

const FEATURES = GROUPS.flatMap((g) =>
  MEASURES.map((m) => ({
    key: `${m.key}_${g.id}`,
    group: g.id,
    label: m.label,
    hint: m.hint,
    range: m.range[g.id],
  }))
);

/* Two real rows from the dataset, so you can try the app straight away. */
const EXAMPLES = {
  benign: [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259],
  malignant: [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
};

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */
const emptyValues = () => Object.fromEntries(FEATURES.map((f) => [f.key, ""]));
const fromArray = (arr) => Object.fromEntries(FEATURES.map((f, i) => [f.key, String(arr[i])]));

/* Turns text into a number, or NaN if it is not a plain non-negative number. */
function toNumber(text) {
  const t = String(text).trim().replace(",", ".");
  if (!/^(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(t)) return NaN;
  const value = Number(t);
  return Number.isFinite(value) ? value : NaN;
}

const fmt = (n) => String(Number(n.toPrecision(4)));

function isOutsideRange(value, [lo, hi]) {
  const pad = 0.02 * (hi - lo);
  return value < lo - pad || value > hi + pad;
}

/* ------------------------------------------------------------------ */
/* Probability gauge: the benign-to-malignant scale with a moving pin. */
/* ------------------------------------------------------------------ */
function Gauge({ probability }) {
  const [pos, setPos] = useState(0.5);

  useEffect(() => {
    const id = requestAnimationFrame(() => setPos(probability));
    return () => cancelAnimationFrame(id);
  }, [probability]);

  const percent = (probability * 100).toFixed(1);

  return (
    <div className="gauge" role="img" aria-label={`Probability of malignancy: ${percent} percent`}>
      <div className="gauge-track">
        <span className="gauge-cutoff" />
        <span className="gauge-pin" style={{ left: `${pos * 100}%` }} />
      </div>
      <div className="gauge-scale" aria-hidden="true">
        <span>Benign</span>
        <span>50% cutoff</span>
        <span>Malignant</span>
      </div>
    </div>
  );
}

function verdictText(p) {
  if (p < 0.3) return "For these measurements the model leans clearly towards benign.";
  if (p > 0.7) return "For these measurements the model leans clearly towards malignant.";
  return "The model is not confident about this one. Only a specialist can say more.";
}

/* ------------------------------------------------------------------ */
/* Main component                                                      */
/* ------------------------------------------------------------------ */
export default function App() {
  const [values, setValues] = useState(emptyValues);
  const [active, setActive] = useState(0);
  const [touched, setTouched] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [server, setServer] = useState("checking"); // checking | online | offline
  const [pasted, setPasted] = useState("");
  const [pasteError, setPasteError] = useState("");
  const tabRefs = useRef([]);
  const resultRef = useRef(null);

  const numbers = useMemo(() => FEATURES.map((f) => toNumber(values[f.key])), [values]);

  const filledPerGroup = useMemo(
    () => GROUPS.map((g) => FEATURES.filter((f, i) => f.group === g.id && !Number.isNaN(numbers[i])).length),
    [numbers]
  );
  const totalFilled = filledPerGroup.reduce((a, b) => a + b, 0);

  /* Check once whether the model server is reachable. */
  useEffect(() => {
    const ctrl = new AbortController();
    fetch(`${API_URL}/health`, { signal: ctrl.signal })
      .then((r) => setServer(r.ok ? "online" : "offline"))
      .catch((e) => {
        if (e.name !== "AbortError") setServer("offline");
      });
    return () => ctrl.abort();
  }, []);

  /* On small screens the result sits below the form, so bring it into view. */
  useEffect(() => {
    if (status !== "done" && status !== "error") return;
    if (!window.matchMedia("(max-width: 880px)").matches) return;
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    resultRef.current?.scrollIntoView({ block: "nearest", behavior: calm ? "auto" : "smooth" });
  }, [status]);

  function resetOutcome() {
    setStatus("idle");
    setResult(null);
    setError("");
  }

  function updateValue(key, text) {
    setValues((v) => ({ ...v, [key]: text }));
    resetOutcome();
  }

  function loadExample(kind) {
    setValues(fromArray(EXAMPLES[kind]));
    setTouched({});
    setSubmitted(false);
    setPasteError("");
    resetOutcome();
  }

  function clearAll() {
    setValues(emptyValues());
    setTouched({});
    setSubmitted(false);
    setPasted("");
    setPasteError("");
    resetOutcome();
  }

  function applyPasted() {
    const parts = pasted.split(/[\s,;]+/).filter(Boolean);
    if (parts.length !== FEATURES.length || parts.some((p) => Number.isNaN(toNumber(p)))) {
      setPasteError(
        `Found ${parts.length} usable values. Paste exactly ${FEATURES.length} numbers, in the order of the tabs, without the id and diagnosis columns.`
      );
      return;
    }
    setValues(fromArray(parts));
    setPasteError("");
    setTouched({});
    setSubmitted(false);
    resetOutcome();
  }

  function focusField(index) {
    const feature = FEATURES[index];
    setActive(GROUPS.findIndex((g) => g.id === feature.group));
    setTimeout(() => document.getElementById(`field-${feature.key.replace(/\s+/g, "-")}`)?.focus(), 0);
  }

  function onTabKeyDown(e) {
    let next = null;
    if (e.key === "ArrowRight") next = (active + 1) % GROUPS.length;
    if (e.key === "ArrowLeft") next = (active - 1 + GROUPS.length) % GROUPS.length;
    if (next === null) return;
    e.preventDefault();
    setActive(next);
    tabRefs.current[next]?.focus();
  }

  async function handlePredict() {
    setSubmitted(true);
    const firstBad = numbers.findIndex((n) => Number.isNaN(n));
    if (firstBad !== -1) {
      resetOutcome();
      focusField(firstBad);
      return;
    }

    const outside = FEATURES.filter((f, i) => isOutsideRange(numbers[i], f.range)).map(
      (f) => `${f.label} (${f.group})`
    );

    setStatus("loading");
    setError("");
    setResult(null);

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 60000);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: numbers }),
        signal: ctrl.signal,
      });

      if (!res.ok) {
        let detail = "";
        try {
          const body = await res.json();
          detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
        } catch {
          /* ignore: the body was not JSON */
        }
        throw new Error(detail || `The server answered with status ${res.status}.`);
      }

      const data = await res.json();
      if (
        typeof data.probability !== "number" ||
        !Number.isFinite(data.probability) ||
        data.probability < 0 ||
        data.probability > 1
      ) {
        throw new Error("The server sent a reply this page cannot read.");
      }

      setResult({
        probability: data.probability,
        label: data.label || (data.probability >= 0.5 ? "Malignant" : "Benign"),
        outside,
      });
      setServer("online");
      setStatus("done");
    } catch (err) {
      let message = err.message;
      if (err.name === "AbortError") {
        message = "The server took too long to answer. Free hosting can need a minute to wake up, so try again shortly.";
        setServer("offline");
      } else if (err instanceof TypeError) {
        message = `Could not reach the model server at ${API_URL}. Check that it is running and that the address is correct.`;
        setServer("offline");
      }
      setError(message);
      setStatus("error");
    } finally {
      clearTimeout(timer);
    }
  }

  const group = GROUPS[active];
  const visible = FEATURES.map((f, i) => ({ f, i })).filter(({ f }) => f.group === group.id);
  const malignant = result?.label === "Malignant";

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1 className="title">Breast cancer classifier</h1>
          <p className="lede">
            Enter 30 measurements taken from a fine-needle biopsy image. A neural network estimates whether the
            tumor is benign or malignant.
          </p>
        </div>
        <p className={`status status-${server}`} role="status">
          <span className="status-dot" aria-hidden="true" />
          {server === "checking" && "Checking model server"}
          {server === "online" && "Model server online"}
          {server === "offline" && "Model server not reachable"}
        </p>
      </header>

      <main className="layout">
        {/* ---------------- Form ---------------- */}
        <section className="form" aria-labelledby="form-title">
          <div className="form-head">
            <h2 id="form-title" className="section-title">Measurements</h2>
            <p className="count">{totalFilled} of {FEATURES.length} filled in</p>
          </div>

          <div className="tabs" role="tablist" aria-label="Measurement groups" onKeyDown={onTabKeyDown}>
            {GROUPS.map((g, i) => (
              <button
                key={g.id}
                ref={(el) => (tabRefs.current[i] = el)}
                type="button"
                role="tab"
                id={`tab-${g.id}`}
                aria-selected={active === i}
                aria-controls={`panel-${g.id}`}
                tabIndex={active === i ? 0 : -1}
                className={`tab${active === i ? " tab-active" : ""}`}
                onClick={() => setActive(i)}
              >
                {g.title}
                <span className="tab-count">{filledPerGroup[i]}/10</span>
              </button>
            ))}
          </div>

          <div role="tabpanel" id={`panel-${group.id}`} aria-labelledby={`tab-${group.id}`} className="panel">
            <p className="blurb">{group.blurb}</p>

            <div className="fields">
              {visible.map(({ f, i }) => {
                const id = `field-${f.key.replace(/\s+/g, "-")}`;
                const text = values[f.key];
                const bad = (touched[f.key] || submitted) && Number.isNaN(numbers[i]);
                const odd = !Number.isNaN(numbers[i]) && isOutsideRange(numbers[i], f.range);
                return (
                  <div className="field" key={f.key}>
                    <label htmlFor={id} className="field-label">{f.label}</label>
                    <input
                      id={id}
                      className={`input${bad ? " input-bad" : ""}${odd ? " input-odd" : ""}`}
                      type="text"
                      inputMode="decimal"
                      autoComplete="off"
                      value={text}
                      placeholder={fmt((f.range[0] + f.range[1]) / 2)}
                      aria-invalid={bad}
                      aria-describedby={`${id}-note`}
                      onChange={(e) => updateValue(f.key, e.target.value)}
                      onBlur={() => setTouched((t) => ({ ...t, [f.key]: true }))}
                      onKeyDown={(e) => e.key === "Enter" && handlePredict()}
                    />
                    <p id={`${id}-note`} className={`note${bad || odd ? " note-warn" : ""}`}>
                      {bad
                        ? "Enter a number, for example 14.5."
                        : odd
                        ? `Outside the range seen in training (${fmt(f.range[0])} to ${fmt(f.range[1])}).`
                        : `${f.hint}. Training range: ${fmt(f.range[0])} to ${fmt(f.range[1])}.`}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>

          <details className="paste">
            <summary>Paste all 30 values at once</summary>
            <label htmlFor="paste-box" className="paste-label">
              Separate the numbers with commas or spaces.
            </label>
            <textarea
              id="paste-box"
              className="paste-box"
              rows={3}
              value={pasted}
              onChange={(e) => setPasted(e.target.value)}
            />
            {pasteError && <p className="note note-warn">{pasteError}</p>}
            <button type="button" className="link-button" onClick={applyPasted}>Fill the form</button>
          </details>

          <div className="actions">
            <div className="actions-links">
              <button type="button" className="link-button" onClick={() => loadExample("benign")}>Load benign example</button>
              <button type="button" className="link-button" onClick={() => loadExample("malignant")}>Load malignant example</button>
              <button type="button" className="link-button" onClick={clearAll}>Clear all</button>
            </div>
            <button type="button" className="primary" onClick={handlePredict} disabled={status === "loading"}>
              {status === "loading" ? "Predicting…" : "Predict"}
            </button>
          </div>
        </section>

        {/* ---------------- Result ---------------- */}
        <aside className="result" ref={resultRef} aria-live="polite" aria-label="Prediction result">
          {status === "idle" && (
            <div className="result-empty">
              <h2 className="section-title">Result</h2>
              <p>Fill in the 30 measurements, or load an example, then press Predict.</p>
              {submitted && totalFilled < FEATURES.length && (
                <p className="note note-warn">
                  {FEATURES.length - totalFilled} fields still need a valid number.
                </p>
              )}
            </div>
          )}

          {status === "loading" && (
            <div className="result-empty">
              <h2 className="section-title">Result</h2>
              <p>Running the model…</p>
            </div>
          )}

          {status === "error" && (
            <div className="result-error" role="alert">
              <h2 className="section-title">Prediction failed</h2>
              <p>{error}</p>
            </div>
          )}

          {status === "done" && result && (
            <div className="result-done">
              <h2 className="section-title">Result</h2>
              <p className={`verdict ${malignant ? "verdict-malignant" : "verdict-benign"}`}>{result.label}</p>
              <p className="probability">
                Probability of malignancy: <strong>{(result.probability * 100).toFixed(1)}%</strong>
              </p>
              <Gauge probability={result.probability} />
              <p className="explain">{verdictText(result.probability)}</p>

              {result.outside.length > 0 && (
                <p className="caution">
                  {result.outside.length} {result.outside.length === 1 ? "value is" : "values are"} outside the range
                  the model learned from (for example {result.outside.slice(0, 3).join(", ")}). Treat this result with
                  extra caution.
                </p>
              )}
            </div>
          )}

          <p className="disclaimer">
            This is a student project for learning. It is not a medical device and cannot give a diagnosis. For any
            health concern, talk to a doctor.
          </p>
        </aside>
      </main>
    </div>
  );
}
