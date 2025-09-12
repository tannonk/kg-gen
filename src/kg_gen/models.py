from pydantic import BaseModel, Field, field_validator
from typing import Tuple, Optional, Dict, Any


# ~~~ DATA STRUCTURES ~~~
class Metadata(BaseModel):
    """Flexible metadata container for graph elements"""

    # All metadata as key-value pairs - no assumptions about specific fields
    data: Dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs):
        """Custom serialization for JSON compatibility"""
        return self.data


class Relation(BaseModel):
    """Enhanced relation with metadata support"""

    subject: str
    predicate: str
    object: str
    metadata: Metadata = Field(default_factory=Metadata)

    class Config:
        # Allow hashing and make it JSON serializable
        frozen = False

    def __hash__(self):
        """Hash based only on knowledge content, not metadata"""
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        """Equality based on semantic content only"""
        if not isinstance(other, Relation):
            return False
        return (self.subject, self.predicate, self.object) == (
            other.subject,
            other.predicate,
            other.object,
        )

    def model_dump(self, **kwargs):
        """Custom serialization for JSON compatibility"""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "metadata": self.metadata.data,
        }

    def merge_metadata(self, other: "Relation") -> "Relation":
        """Create new relation with merged metadata from both relations"""
        if not isinstance(other, Relation):
            return self

        # Merge all metadata, with special handling for source field
        merged_data = {**self.metadata.data}

        for key, value in other.metadata.data.items():
            if key == "source" and key in merged_data:
                # Special handling for source field - combine them
                merged_data[key] = self._merge_sources(merged_data[key], value)
            elif key not in merged_data:
                merged_data[key] = value
            # If key exists and it's not source, keep the existing value (first wins)

        merged = Metadata(data=merged_data)

        return Relation(
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
            metadata=merged,
        )

    def _merge_sources(
        self, source1: Optional[str], source2: Optional[str]
    ) -> Optional[list[str]]:
        """Merge source metadata - combine into comma-separated list"""
        sources = [s for s in [source1, source2] if s is not None]
        if not sources:
            return None
        # Deduplicate and sort for consistent ordering
        unique_sources = sorted(set(sources))
        return unique_sources


class Graph(BaseModel):
    entities: set[str] = Field(
        ..., description="All entities including additional ones from response"
    )
    edges: set[str] = Field(..., description="All edges")
    relations: set[Relation] = Field(..., description="List of relations with metadata")
    entity_clusters: Optional[dict[str, set[str]]] = None
    edge_clusters: Optional[dict[str, set[str]]] = None
    # Optional graph-level metadata
    graph_metadata: Optional[Metadata] = None

    @field_validator("relations", mode="before")
    @classmethod
    def convert_tuple_relations(cls, v):
        """Convert old tuple format relations to new Relation objects for backward compatibility"""
        if not v:
            return v

        # If it's already a set of Relation objects, pass through
        if isinstance(v, set) and all(isinstance(r, Relation) for r in v if r):
            return v

        # Convert tuples/lists/dicts to Relation objects
        converted = set()

        # Handle both list and set inputs
        items = v if hasattr(v, "__iter__") else [v]

        for item in items:
            if isinstance(item, (tuple, list)) and len(item) == 3:
                # Convert (subject, predicate, object) to Relation with default metadata
                converted.add(
                    Relation(
                        subject=str(item[0]),
                        predicate=str(item[1]),
                        object=str(item[2]),
                        metadata=Metadata(),  # Default empty metadata
                    )
                )
            elif isinstance(item, Relation):
                converted.add(item)
            elif (
                isinstance(item, dict)
                and "subject" in item
                and "predicate" in item
                and "object" in item
            ):
                # Handle dict format (from JSON deserialization)
                metadata_data = item.get("metadata", {})
                if isinstance(metadata_data, dict):
                    metadata = Metadata(data=metadata_data)
                else:
                    metadata = Metadata()
                converted.add(
                    Relation(
                        subject=str(item["subject"]),
                        predicate=str(item["predicate"]),
                        object=str(item["object"]),
                        metadata=metadata,
                    )
                )
            else:
                # If we can't convert, let Pydantic handle the validation error
                return v

        return converted

    def get_relation_tuples(self) -> set[Tuple[str, str, str]]:
        """Backward compatibility: convert relations to tuples"""
        return {(r.subject, r.predicate, r.object) for r in self.relations}

    def model_dump(self, **kwargs):
        """Custom serialization to handle sets for JSON serialization"""
        # Skip super().model_dump() to avoid unhashable dict issues with sets of Relations
        # Manually construct the serializable dictionary

        return {
            "entities": list(self.entities),
            "edges": list(self.edges),
            "relations": [
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "metadata": r.metadata.data,
                }
                for r in self.relations
            ],
            "entity_clusters": {k: list(v) for k, v in self.entity_clusters.items()}
            if self.entity_clusters
            else None,
            "edge_clusters": {k: list(v) for k, v in self.edge_clusters.items()}
            if self.edge_clusters
            else None,
            "graph_metadata": self.graph_metadata.data if self.graph_metadata else None,
        }
