import { useState, useEffect, useRef } from "react";
import { I } from "../utils/icons";

// ═══════════════════════════════════════════════════════════════════════
// UI Components (extracted from CognithorControlCenter.jsx)
// ═══════════════════════════════════════════════════════════════════════

export function Toggle({ label, value, onChange, desc }) {
  return (
    <div className="cc-field">
      <div className="cc-field-row" onClick={() => onChange(!value)} style={{ cursor: "pointer" }}>
        <div>
          <div className="cc-label">{label}</div>
          {desc && <div className="cc-desc">{desc}</div>}
        </div>
        <div className={`cc-toggle ${value ? "on" : ""}`}><div className="cc-toggle-dot" /></div>
      </div>
    </div>
  );
}

// Fix #7: validation prop for inputs + tooltip support
export function TextInput({ label, value, onChange, desc, placeholder, type = "text", mono, error, disabled, tooltip }) {
  const [show, setShow] = useState(false);
  const isSecret = type === "password";
  return (
    <div className="cc-field">
      <div className="cc-label">{label} {tooltip && <span className="cc-tooltip-trigger" title={tooltip}>{I.help}</span>}</div>
      {desc && <div className="cc-desc">{desc}</div>}
      <div className="cc-input-wrap">
        <input
          className={`cc-input ${mono ? "mono" : ""} ${error ? "cc-error" : ""} ${disabled ? "cc-input-disabled" : ""}`}
          type={isSecret && !show ? "password" : "text"}
          value={value || ""}
          onChange={e => !disabled && onChange(e.target.value)}
          placeholder={placeholder || ""}
          readOnly={disabled}
          tabIndex={disabled ? -1 : 0}
          aria-label={label}
          aria-invalid={!!error}
        />
        {isSecret && (
          <button className="cc-eye-btn" onClick={() => setShow(!show)} type="button" aria-label={show ? "Verbergen" : "Anzeigen"}>{show ? I.eyeOff : I.eye}</button>
        )}
      </div>
      {error && <div className="cc-field-error" role="alert">{error}</div>}
    </div>
  );
}

export function NumberInput({ label, value, onChange, desc, min, max, step = 1, error }) {
  const localErr = (value !== undefined && value !== null) && ((min !== undefined && value < min) || (max !== undefined && value > max));
  const displayErr = error || (localErr ? `Wert muss zwischen ${min} und ${max} liegen.` : null);
  return (
    <div className="cc-field">
      <div className="cc-label">{label}</div>
      {desc && <div className="cc-desc">{desc}</div>}
      <input
        className={`cc-input ${displayErr ? "cc-error" : ""}`}
        type="number"
        value={value ?? ""}
        onChange={e => onChange(e.target.value === "" ? null : Number(e.target.value))}
        min={min} max={max} step={step}
      />
      {displayErr && <div className="cc-field-error">{displayErr}</div>}
    </div>
  );
}

// Fix #21: Slider with editable value
export function SliderInput({ label, value, onChange, min = 0, max = 1, step = 0.01, desc }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const commit = () => {
    const n = parseFloat(draft);
    if (!isNaN(n) && n >= min && n <= max) onChange(n);
    setEditing(false);
  };
  return (
    <div className="cc-field">
      <div className="cc-field-row">
        <div>
          <div className="cc-label">{label}</div>
          {desc && <div className="cc-desc">{desc}</div>}
        </div>
        {editing ? (
          <input
            className="cc-slider-edit"
            autoFocus
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={e => { if (e.key === "Enter") commit(); if (e.key === "Escape") setEditing(false); }}
          />
        ) : (
          <span
            className="cc-slider-val"
            onClick={() => { setDraft(typeof value === "number" ? value.toFixed(2) : String(value)); setEditing(true); }}
            title="Klicken zum manuellen Eingeben"
          >
            {typeof value === "number" ? value.toFixed(2) : value}
          </span>
        )}
      </div>
      <input type="range" className="cc-slider" value={value ?? min} onChange={e => onChange(Number(e.target.value))} min={min} max={max} step={step} />
    </div>
  );
}

export function SelectInput({ label, value, onChange, options, desc }) {
  return (
    <div className="cc-field">
      <div className="cc-label">{label}</div>
      {desc && <div className="cc-desc">{desc}</div>}
      <select className="cc-select" value={value || ""} onChange={e => onChange(e.target.value)}>
        {options.map(o => typeof o === "string" ? <option key={o} value={o}>{o}</option> : <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

export function ListInput({ label, value = [], onChange, desc, placeholder }) {
  const [draft, setDraft] = useState("");
  const add = () => { if (draft.trim()) { onChange([...value, draft.trim()]); setDraft(""); } };
  return (
    <div className="cc-field">
      <div className="cc-label">{label}</div>
      {desc && <div className="cc-desc">{desc}</div>}
      <div className="cc-list-items">
        {value.map((v, i) => (
          <div key={i} className="cc-list-item">
            <span className="mono">{v}</span>
            <button className="cc-btn-icon" onClick={() => onChange(value.filter((_, j) => j !== i))} type="button">{I.trash}</button>
          </div>
        ))}
      </div>
      <div className="cc-list-add">
        <input className="cc-input" value={draft} onChange={e => setDraft(e.target.value)} placeholder={placeholder || "Hinzuf\u00fcgen\u2026"} onKeyDown={e => e.key === "Enter" && add()} />
        <button className="cc-btn-sm" onClick={add} type="button">{I.plus}</button>
      </div>
    </div>
  );
}

// Fix #8: JSON editor with validation + Fix #9: reset button
export function TextArea({ label, value, onChange, desc, rows = 8, mono, error, onReset, resetLabel }) {
  return (
    <div className="cc-field">
      <div className="cc-field-row">
        <div>
          <div className="cc-label">{label}</div>
          {desc && <div className="cc-desc">{desc}</div>}
        </div>
        {onReset && (
          <button className="cc-btn-reset" onClick={onReset} title={resetLabel || "Auf Standard zur\u00fccksetzen"} type="button">
            {I.reset} <span>Reset</span>
          </button>
        )}
      </div>
      <textarea
        className={`cc-textarea ${mono ? "mono" : ""} ${error ? "cc-error" : ""}`}
        rows={rows}
        value={value || ""}
        onChange={e => onChange(e.target.value)}
      />
      {error && <div className="cc-field-error">{error}</div>}
    </div>
  );
}

// Fix #8: Dedicated JSON textarea with live validation
// B7: Prevents cursor-jump by not syncing raw from parent during active editing
export function JsonEditor({ label, value, onChange, desc, rows = 6, onValidationError }) {
  const [raw, setRaw] = useState(() => typeof value === "string" ? value : JSON.stringify(value || {}, null, 2));
  const [err, setErr] = useState(null);
  const editingRef = useRef(false);
  const onValidationErrorRef = useRef(onValidationError);

  useEffect(() => {
    onValidationErrorRef.current = onValidationError;
  }, [onValidationError]);

  useEffect(() => {
    // Only sync from parent when NOT actively editing (e.g. external reset)
    if (!editingRef.current) {
      setRaw(typeof value === "string" ? value : JSON.stringify(value || {}, null, 2));
      setErr(null);
      if (onValidationErrorRef.current) onValidationErrorRef.current(null);
    }
  }, [value]);

  const handleChange = (txt) => {
    editingRef.current = true;
    setRaw(txt);
    try {
      const parsed = JSON.parse(txt);
      setErr(null);
      if (onValidationErrorRef.current) onValidationErrorRef.current(null);
      onChange(parsed);
    } catch (e) {
      const errorMsg = `JSON-Fehler: ${e.message.replace(/^JSON\.parse: /, "")}`;
      setErr(errorMsg);
      if (onValidationErrorRef.current) onValidationErrorRef.current(errorMsg);
    }
    // Reset editing flag after a short debounce
    clearTimeout(editingRef._timer);
    editingRef._timer = setTimeout(() => { editingRef.current = false; }, 500);
  };
  return (
    <TextArea
      label={label}
      value={raw}
      onChange={handleChange}
      desc={desc}
      rows={rows}
      mono
      error={err}
    />
  );
}

// Fix #3: Read-only info display
export function ReadOnly({ label, value, desc }) {
  return (
    <div className="cc-field">
      <div className="cc-label">{label}</div>
      {desc && <div className="cc-desc">{desc}</div>}
      <div className="cc-readonly">{value || "\u2014"}</div>
    </div>
  );
}

// Fix #12: Card that can be externally forced open
export function Card({ title, children, open: initOpen = true, badge, forceOpen }) {
  const [open, setOpen] = useState(initOpen);
  const prevForce = useRef(forceOpen);
  useEffect(() => {
    if (forceOpen && !prevForce.current) setOpen(true);
    prevForce.current = forceOpen;
  }, [forceOpen]);
  return (
    <div className="cc-card">
      <div className="cc-card-head" onClick={() => setOpen(!open)}>
        <span className="cc-card-title">{title}</span>
        <div className="cc-card-right">
          {badge && <span className={`cc-badge ${badge}`}>{badge}</span>}
          <span className={`cc-chevron ${open ? "open" : ""}`}>{"\u25BE"}</span>
        </div>
      </div>
      {open && <div className="cc-card-body">{children}</div>}
    </div>
  );
}

export function Section({ title, desc }) {
  return (
    <div className="cc-section-head">
      <h2 className="cc-section-title">{title}</h2>
      {desc && <p className="cc-section-desc">{desc}</p>}
    </div>
  );
}

// Fix #5: Loading spinner
export function Spinner() {
  return (
    <div className="cc-spinner-wrap">
      <div className="cc-spinner" />
      <span className="cc-spinner-text">Konfiguration wird geladen\u2026</span>
    </div>
  );
}
