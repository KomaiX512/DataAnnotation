from template.hazard.serving import CommercialServingGateway, PromotionRegistry
from template.hazard.vector_db import OshaVectorDatabase


def test_osha_vector_database_returns_ranked_refs():
    db = OshaVectorDatabase.default()
    refs = db.search("fall protection harness edge risk", top_k=2)
    assert len(refs) == 2
    assert refs[0].citation_id.startswith("29CFR")


def test_promotion_registry_and_gateway():
    registry = PromotionRegistry(min_promotion_score=0.7)
    promoted = registry.maybe_promote(
        uid=3,
        model_hash="model-hash-xyz",
        score=0.82,
        step=12,
    )
    assert promoted is True
    gateway = CommercialServingGateway(registry)
    assert gateway.select_model_hash() == "model-hash-xyz"
