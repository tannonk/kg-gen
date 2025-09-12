import logging
from ..models import Graph, Relation
import dspy
from typing import Optional, Literal
from pydantic import BaseModel
from ..utils.logging_config import setup_logger, log_operation
from ..utils.usage_tracker import usage_tracker
import mlflow

dspy.enable_logging()
logging.getLogger("dspy").setLevel(logging.DEBUG)

LOOP_N = 8
BATCH_SIZE = 10

ItemType = Literal["entities", "edges"]


class ChooseRepresentative(dspy.Signature):
    """Select the best item name to represent the cluster, ideally from the cluster.
    Prefer shorter names and generalizability across the cluster."""

    cluster: set[str] = dspy.InputField()
    context: str = dspy.InputField(desc="the larger context in which the items appear")
    representative: str = dspy.OutputField()


choose_rep = dspy.Predict(ChooseRepresentative)


class Cluster(BaseModel):
    representative: str
    members: set[str]


@mlflow.trace
def cluster_items(
    dspy,
    items: set[str],
    item_type: ItemType = "entities",
    context: str = "",
    logger: logging.Logger = None,
) -> tuple[set[str], dict[str, set[str]]]:
    """Returns item set and cluster dict mapping representatives to sets of items"""

    if logger is None:
        logger = setup_logger(f"kg_gen.clustering.{item_type}")
    logger.info(f"Starting {item_type} clustering with {len(items)} items")
    logger.debug(f"Context: {context}")

    context = f"{item_type.upper()} of a graph extracted from source text. {context}"
    remaining_items = items.copy()
    clusters: list[Cluster] = []
    no_progress_count = 0

    logger.debug(f"Starting iterative clustering with max {LOOP_N} loops")

    while len(remaining_items) > 0:
        logger.debug(
            f"Clustering iteration: {len(remaining_items)} items remaining, {len(clusters)} clusters formed"
        )

        ItemsLiteral = Literal[tuple(items)]

        class ExtractCluster(dspy.Signature):
            """Find one cluster of related items from the list.
            A cluster should contain items that are the same in meaning, with different tenses, plural forms, stem forms, or cases.
            Return populated list only if you find items that clearly belong together, else return empty list."""

            items: set[ItemsLiteral] = dspy.InputField()
            context: str = dspy.InputField(
                desc="The larger context in which the items appear"
            )
            cluster: list[ItemsLiteral] = dspy.OutputField()

        extract = dspy.Predict(ExtractCluster)

        extract_result = extract(items=remaining_items, context=context)

        # Track usage with the global usage tracker
        usage_tracker.track_usage(extract_result, step="ExtractCluster", logger=logger)

        suggested_cluster: set[ItemsLiteral] = set(extract_result.cluster)

        if len(suggested_cluster) > 0:
            ClusterLiteral = Literal[tuple(suggested_cluster)]

            class ValidateCluster(dspy.Signature):
                """Validate if these items belong in the same cluster.
                A cluster should contain items that are the same in meaning, with different tenses, plural forms, stem forms, or cases.
                Return populated list only if you find items that clearly belong together, else return empty list."""

                cluster: set[ClusterLiteral] = dspy.InputField()
                context: str = dspy.InputField(
                    desc="The larger context in which the items appear"
                )
                validated_items: list[ClusterLiteral] = dspy.OutputField(
                    desc="All the items that belong together in the cluster"
                )

            validate = dspy.Predict(ValidateCluster)

            validate_result = validate(cluster=suggested_cluster, context=context)

            # Track usage with the global usage tracker
            usage_tracker.track_usage(
                validate_result, step="ValidateCluster", logger=logger
            )

            validated_cluster: set[ItemsLiteral] = set(validate_result.validated_items)

            if len(validated_cluster) > 1:
                no_progress_count = (
                    0  # Reset no-progress counter on successful clustering
                )

                representative_result = choose_rep(
                    cluster=validated_cluster, context=context
                )
                usage_tracker.track_usage(
                    representative_result, step="ChooseRepresentative", logger=logger
                )
                representative = representative_result.representative

                clusters.append(
                    Cluster(representative=representative, members=validated_cluster)
                )
                remaining_items = {
                    item for item in remaining_items if item not in validated_cluster
                }

                logger.debug(
                    f"Created cluster '{representative}' with {len(validated_cluster)} members: {list(validated_cluster)}"
                )
                continue

        no_progress_count += 1

        if no_progress_count >= LOOP_N or len(remaining_items) == 0:
            logger.debug(
                "No progress in clustering or no remaining items, moving to batch processing"
            )
            break

    if len(remaining_items) > 0:
        logger.debug(
            f"Processing {len(remaining_items)} remaining items in batches of {BATCH_SIZE}"
        )
        items_to_process = list(remaining_items)

        for i in range(0, len(items_to_process), BATCH_SIZE):
            batch = items_to_process[i : min(i + BATCH_SIZE, len(items_to_process))]
            BatchLiteral = Literal[tuple(batch)]

            if not clusters:
                for item in batch:
                    clusters.append(Cluster(representative=item, members={item}))
                continue

            class CheckExistingClusters(dspy.Signature):
                """Determine if the given items can be added to any of the existing clusters.
                Return representative of matching cluster for each item, or None if there is no match."""

                items: list[BatchLiteral] = dspy.InputField()
                clusters: list[Cluster] = dspy.InputField(
                    desc="Mapping of cluster representatives to their cluster members"
                )
                context: str = dspy.InputField(
                    desc="The larger context in which the items appear"
                )
                cluster_reps_that_items_belong_to: list[Optional[str]] = (
                    dspy.OutputField(
                        desc="Ordered list of cluster representatives where each is the cluster where that item belongs to, or None if no match. THIS LIST LENGTH IS SAME AS ITEMS LIST LENGTH"
                    )
                )

            check_existing = dspy.ChainOfThought(CheckExistingClusters)

            c_result = check_existing(items=batch, clusters=clusters, context=context)

            # Track usage with the global usage tracker
            usage_tracker.track_usage(
                c_result, step="CheckExistingClusters", logger=logger
            )

            cluster_reps = c_result.cluster_reps_that_items_belong_to

            # Map representatives to their cluster objects for easier lookup
            # Ensure cluster_map uses the most up-to-date list of clusters
            cluster_map = {c.representative: c for c in clusters}

            # Determine assignments for batch items based on validation
            # Stores item -> assigned representative. If None, item needs a new cluster.
            item_assignments: dict[str, Optional[str]] = {}

            for i, item in enumerate(batch):
                # Default: item might become its own cluster if no valid assignment found
                item_assignments[item] = None

                # Get the suggested representative from the LLM call
                rep = cluster_reps[i] if i < len(cluster_reps) else None

                target_cluster = None
                # Check if the suggested representative corresponds to an existing cluster
                if rep is not None and rep in cluster_map:
                    target_cluster = cluster_map[rep]

                if target_cluster:
                    # If the item is already the representative or a member, assign it definitively
                    if (
                        item == target_cluster.representative
                        or item in target_cluster.members
                    ):
                        item_assignments[item] = target_cluster.representative
                        continue  # Move to the next item

                    # Validate adding the item to the existing cluster's members
                    potential_new_members = target_cluster.members | {item}
                    try:
                        # Call the validation signature
                        v_result = validate(
                            cluster=potential_new_members, context=context
                        )
                        validated_items = set(
                            v_result.validated_items
                        )  # Ensure result is a set

                        # Check if the item was validated as part of the cluster AND
                        # the size matches the expected size after adding.
                        # This assumes 'validate' confirms membership without removing others.
                        if item in validated_items and len(validated_items) == len(
                            potential_new_members
                        ):
                            # Validation successful, assign item to this cluster's representative
                            item_assignments[item] = target_cluster.representative
                        # Else: Validation failed or item rejected, item_assignments[item] remains None

                    except Exception as e:
                        # Handle potential errors during the validation call
                        logger.warning(
                            f"Validation failed for item '{item}' potentially belonging to cluster '{target_cluster.representative}': {e}"
                        )
                        # Keep item_assignments[item] as None, indicating it needs a new cluster

                # Else (no valid target_cluster found for the suggested 'rep'):
                # item_assignments[item] remains None, will become a new cluster.

            # Process the assignments determined above
            new_cluster_items = set()  # Collect items needing a brand new cluster
            for item, assigned_rep in item_assignments.items():
                if assigned_rep is not None:
                    # Item belongs to an existing cluster, add it to the members set
                    # Ensure the cluster exists in the map (should always be true here)
                    if assigned_rep in cluster_map:
                        cluster_map[assigned_rep].members.add(item)
                    else:
                        # This case should ideally not happen if logic is correct
                        logger.error(
                            f"Error: Assigned representative '{assigned_rep}' not found in cluster_map for item '{item}'. Creating new cluster."
                        )
                        if (
                            item not in cluster_map
                        ):  # Avoid creating if item itself is already a rep
                            new_cluster_items.add(item)
                else:
                    # Item needs a new cluster, unless it's already a representative itself
                    if item not in cluster_map:
                        new_cluster_items.add(item)

            # Create the new Cluster objects for items that couldn't be assigned
            for item in new_cluster_items:
                # Final check: ensure a cluster with this item as rep doesn't exist
                if item not in cluster_map:
                    new_cluster = Cluster(representative=item, members={item})
                    clusters.append(new_cluster)
                    cluster_map[item] = (
                        new_cluster  # Update map for internal consistency
                    )

    # Prepare the final output format expected by the calling function:
    # 1. A dictionary mapping representative -> set of members
    # 2. A set containing all unique representatives
    final_clusters_dict = {c.representative: c.members for c in clusters}
    new_items = set(final_clusters_dict.keys())  # The set of representatives

    # Log clustering results
    total_clustered = sum(len(members) for members in final_clusters_dict.values())
    compression_ratio = total_clustered / len(items) if len(items) > 0 else 0

    logger.info(
        f"Clustering completed: {len(items)} -> {len(new_items)} items "
        f"({len(final_clusters_dict)} clusters, {compression_ratio:.1%} compression)"
    )

    return new_items, final_clusters_dict


@mlflow.trace
@log_operation("Graph Clustering")
def cluster_graph(
    dspy,
    graph: Graph,
    context: str = "",
    entity_cluster_context: str = "",
    relation_cluster_context: str = "",
    skip_entity_clustering: bool = False,
    skip_relation_clustering: bool = False,
    logger: logging.Logger = None,
) -> Graph:
    """Cluster entities and edges in a graph, updating relations accordingly.

    Args:
        dspy: The DSPy runtime
        graph: Input graph with entities, edges, and relations
        context: Additional context string for clustering
        logger: Logger instance to use for logging

    Returns:
        Graph with clustered entities and edges, updated relations, and cluster mappings
    """

    if logger is None:
        logger = setup_logger("kg_gen.clustering")

    logger.info(
        f"Starting graph clustering: {len(graph.entities)} entities, {len(graph.edges)} edges, {len(graph.relations)} relations"
    )

    if not skip_entity_clustering:
        if entity_cluster_context:
            context = entity_cluster_context
            logger.debug(
                f"Using entity-specific context for entity clustering: {context}"
            )
        entities, entity_clusters = cluster_items(
            dspy, graph.entities, "entities", context, logger=logger
        )
    else:
        entities, entity_clusters = graph.entities, {e: {e} for e in graph.entities}
        logger.info("Skipping entity clustering as per configuration")

    if not skip_relation_clustering:
        if relation_cluster_context:
            context = relation_cluster_context
            logger.debug(
                f"Using relation-specific context for relation clustering: {context}"
            )
        edges, edge_clusters = cluster_items(dspy, graph.edges, "edges", context, logger=logger)
    else:
        edges, edge_clusters = graph.edges, {e: {e} for e in graph.edges}
        logger.info("Skipping edge clustering as per configuration")

    # Update relations based on clusters
    logger.debug("Updating relations based on entity and edge clusters")
    relations: set[Relation] = set()

    for relation in graph.relations:
        s, p, o = relation.subject, relation.predicate, relation.object

        # Look up subject in entity clusters
        if s not in entities:
            for rep, cluster in entity_clusters.items():
                if s in cluster:
                    s = rep
                    break

        # Look up predicate in edge clusters
        if p not in edges:
            for rep, cluster in edge_clusters.items():
                if p in cluster:
                    p = rep
                    break

        # Look up object in entity clusters
        if o not in entities:
            for rep, cluster in entity_clusters.items():
                if o in cluster:
                    o = rep
                    break

        # Create new relation with clustered components but preserve original metadata
        clustered_relation = Relation(
            subject=s,
            predicate=p,
            object=o,
            metadata=relation.metadata,  # Preserve original metadata
        )
        relations.add(clustered_relation)

    return Graph(
        entities=entities,
        edges=edges,
        relations=relations,
        entity_clusters=entity_clusters,
        edge_clusters=edge_clusters,
    )


if __name__ == "__main__":
    import os
    from ..kg_gen import KGGen

    logger = setup_logger("kg_gen.clustering.example", "INFO", True)

    model = "openai/gpt-4o"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Please set OPENAI_API_KEY environment variable")
        exit(1)

    # Example with family relationships
    kg_gen = KGGen(model=model, temperature=0.0, api_key=api_key)
    graph = Graph(
        entities={"Linda", "Joshua", "Josh", "Ben", "Andrew", "Judy"},
        edges={
            "is mother of",
            "is brother of",
            "is father of",
            "is sister of",
            "is nephew of",
            "is aunt of",
            "is same as",
        },
        relations={
            ("Linda", "is mother of", "Joshua"),
            ("Ben", "is brother of", "Josh"),
            ("Andrew", "is father of", "Josh"),
            ("Judy", "is sister of", "Andrew"),
            ("Josh", "is nephew of", "Judy"),
            ("Judy", "is aunt of", "Josh"),
            ("Josh", "is same as", "Joshua"),
        },
    )

    try:
        clustered_graph = kg_gen.cluster(graph=graph)
        logger.info(f"Clustered graph: {clustered_graph}")

    except Exception as e:
        raise ValueError(e)
