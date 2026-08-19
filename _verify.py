from aic2026.agent.tools import RetrievalTools
from aic2026.reranking.color import set_colour_sidecar, _COLOUR_SIDECAR_PATH
t = RetrievalTools.__dataclass_fields__
print("colour_sidecar_path default:", t["colour_sidecar_path"].default)
print("colour_rerank_weight default:", t["colour_rerank_weight"].default)
print("verify OK")
