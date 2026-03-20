# Cognithor Helm Chart

Deploy Cognithor Agent OS on Kubernetes with an optional Ollama sidecar.

## Installation

```bash
helm install cognithor ./deploy/helm/cognithor
```

With custom values:

```bash
helm install cognithor ./deploy/helm/cognithor -f my-values.yaml
```

## Configuration Reference

| Parameter | Description | Default |
|---|---|---|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image | `ghcr.io/alex8791-cyber/cognithor` |
| `image.tag` | Image tag | `0.48.0` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `8741` |
| `ingress.enabled` | Enable ingress | `false` |
| `ingress.className` | Ingress class | `nginx` |
| `ollama.enabled` | Deploy Ollama sidecar | `true` |
| `ollama.gpu.enabled` | Enable GPU for Ollama | `false` |
| `ollama.gpu.count` | Number of GPUs | `1` |
| `ollama.models` | Models to pull on init | `[qwen3:8b, nomic-embed-text]` |
| `persistence.data.enabled` | Persistent data volume | `true` |
| `persistence.data.size` | Data volume size | `10Gi` |
| `persistence.ollama.size` | Ollama models volume size | `50Gi` |
| `config.language` | UI/response language | `en` |
| `config.llm_backend_type` | LLM backend | `ollama` |
| `config.operation_mode` | Agent operation mode | `autonomous` |
| `config.extra` | Additional config.yaml keys | `{}` |

## GPU Setup

To run Ollama with GPU acceleration:

1. Install the NVIDIA device plugin for Kubernetes.
2. Enable GPU in values:

```yaml
ollama:
  gpu:
    enabled: true
    count: 1
```

This adds `nvidia.com/gpu` resource limits and a `nvidia.com/gpu.present` nodeSelector.

## Ingress with WebSocket Support

The chart includes WebSocket-compatible nginx annotations by default. To enable:

```yaml
ingress:
  enabled: true
  className: nginx
  hosts:
    - host: cognithor.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: cognithor-tls
      hosts:
        - cognithor.example.com
```

## Scaling Considerations

Cognithor is stateful (SQLite databases, local caches). Running multiple replicas requires:

- Shared or replicated persistent storage
- Sticky sessions if using multiple replicas behind a load balancer

For most deployments, a single replica with adequate resources is recommended. Scale vertically (more CPU/memory) rather than horizontally.

## Uninstall

```bash
helm uninstall cognithor
```

Note: PVCs are not deleted automatically. To remove all data:

```bash
kubectl delete pvc -l app.kubernetes.io/name=cognithor
```
