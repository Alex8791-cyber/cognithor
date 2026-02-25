import { FunctionComponent } from 'preact';

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

const ApprovalDialog: FunctionComponent<ApprovalDialogProps> = ({
  request,
  onApprove,
  onReject,
}) => {
  return (
    <div class="approval-overlay">
      <div class="approval-dialog">
        <h3>Genehmigung erforderlich</h3>
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
              <pre>{request.params}</pre>
            </div>
          )}
        </div>
        <div class="approval-actions">
          <button class="btn-approve" onClick={() => onApprove(request.id)}>
            Genehmigen
          </button>
          <button class="btn-reject" onClick={() => onReject(request.id)}>
            Ablehnen
          </button>
        </div>
      </div>
    </div>
  );
};

export default ApprovalDialog;
