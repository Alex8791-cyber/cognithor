import { useEffect, useRef, useCallback } from "react";

/**
 * Styled Confirm-Modal — replaces native confirm() dialogs.
 */
export function ConfirmModal({
  open,
  title,
  message,
  confirmLabel = "Bestätigen",
  cancelLabel = "Abbrechen",
  danger = false,
  onConfirm,
  onCancel,
}) {
  const dialogRef = useRef(null);

  useEffect(() => {
    if (open && dialogRef.current) {
      dialogRef.current.focus();
    }
  }, [open]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === "Escape") onCancel();
    if (e.key === "Enter") onConfirm();
  }, [onCancel, onConfirm]);

  if (!open) return null;

  return (
    <div className="cc-modal-overlay" onClick={onCancel} role="dialog" aria-modal="true" aria-label={title}>
      <div
        className="cc-modal"
        ref={dialogRef}
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <h3 className="cc-modal-title">{title}</h3>
        <p className="cc-modal-message">{message}</p>
        <div className="cc-modal-actions">
          <button className="cc-modal-btn cc-modal-btn-cancel" onClick={onCancel}>
            {cancelLabel}
          </button>
          <button
            className={`cc-modal-btn ${danger ? "cc-modal-btn-danger" : "cc-modal-btn-confirm"}`}
            onClick={onConfirm}
            autoFocus
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
