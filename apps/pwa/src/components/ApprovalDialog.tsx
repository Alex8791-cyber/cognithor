interface ApprovalRequest {
  id: string;
  tool: string;
  reason: string;
  params: string;
}

interface ApprovalDialogProps {
  request: ApprovalRequest;
  onApprove: (id: string) => void;
  onReject: (id: string) => void;
}

export function ApprovalDialog({ request, onApprove, onReject }: ApprovalDialogProps) {
  return (
    <div class="approval-overlay" role="dialog" aria-modal="true" aria-label="Genehmigung erforderlich">
      <div class="approval-dialog">
        <h3 class="approval-title">Genehmigung erforderlich</h3>
        <div class="approval-details">
          <div class="approval-field">
            <strong>Tool:</strong>
            <code>{request.tool}</code>
          </div>
          <div class="approval-field">
            <strong>Grund:</strong>
            <span>{request.reason}</span>
          </div>
          {request.params && (
            <div class="approval-field">
              <strong>Parameter:</strong>
              <pre class="approval-params">{request.params}</pre>
            </div>
          )}
        </div>
        <div class="approval-actions">
          <button
            class="approval-btn approval-btn-approve"
            onClick={() => onApprove(request.id)}
            aria-label="Genehmigen"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M20 6L9 17l-5-5" />
            </svg>
            Genehmigen
          </button>
          <button
            class="approval-btn approval-btn-reject"
            onClick={() => onReject(request.id)}
            aria-label="Ablehnen"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
            Ablehnen
          </button>
        </div>
      </div>
    </div>
  );
}
