---
title: Patched `PYTORCH_DISTRIBUTIONS` is no longer correct
assignees: pmeier
---

The variable `light_the_torch._patch.PYTORCH_DISTRIBUTIONS` is no longer aligned with
the wheels hosted by PyTorch. Please replace it with

```py
PYTORCH_DISTRIBUTIONS = {
{%- for dist in env.PYTORCH_DISTRIBUTIONS %}
    "{{ dist }}",
{%- endfor %}
}
```
