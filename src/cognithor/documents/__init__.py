"""Documents package: template management for Typst-based document generation.

**Stability:** feature-complete. Used by `cognithor.mcp.media` for invoice and
report generation. Expand only when adding new template formats; the public
surface (`DocumentTemplate`, `TemplateManager`) is considered stable API.
"""

from cognithor.documents.templates import DocumentTemplate, TemplateManager

__all__ = ["DocumentTemplate", "TemplateManager"]
