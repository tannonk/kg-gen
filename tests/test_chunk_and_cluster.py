from kg_gen import KGGen
import os
from pathlib import Path
from dotenv import load_dotenv

from tests.utils.visualize_kg import visualize

import mlflow

# Tell MLflow about the server URI.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("test_chunk_and_cluster")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Example usage
    # model = "openai/microsoft/Phi-4-mini-instruct"
    # model="openai/Qwen/Qwen3-8B-AWQ",
    model = "openai/gpt-4.1-nano-2025-04-14"
    kg = KGGen(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://api.openai.com/v1",
        # api_key="EMPTY",
        # api_base="http://localhost:8000/v1",
        max_tokens=8000,
        log_level="DEBUG",
        force=True,  # must be true to ensure accurate cost tracking
    )

    # Load fresh wiki content
    # with open('tests/data/fresh_wiki_article.md', 'r', encoding='utf-8') as f:
    #   text = f.read()
    # text = """Recently, in addition to logging residues, stumps have become an important component in energy production since there is growing global interest in the use of renewable energy sources in order to decrease anthropogenic carbon emissions. Harvesting of stumps influences the forest floor by changing vegetation and soil organic layers and exposing mineral soil across large areas. We studied whether stump harvesting after clear felling poses further short-term changes in boreal forest soil decomposer community (microbes and mesofauna) and vegetation when compared to the traditional site preparation practice (mounding). In general, stump harvesting caused decline in enchytraeid abundance but did not induce further major changes in decomposer community otherwise nor in vegetation of each soil micro-habitat (intact soil and exposed mineral soil). However, the abundances of almost all decomposer animals were lower in the exposed mineral soil than in the intact soil. Stump removal increased the area of exposed mineral soil in the clear felled areas, leading to lower amount of high quality habitat for most decomposer organisms. Hence, it is obvious that there are (or will be) differences in the decomposer community dynamics between the treatments at the forest stand level. Both species richness and coverage of plants benefitted from large-scale exposure of mineral soil. Because the stump removal procedure disturbs soil organic layers and negatively affects the decomposer community, it has the potential to alter nutrient dynamics in forests."""
    # text = """**Suppression of the hazardous substances in catalytically upgraded bio-heavy oil as a precautious measure for clean air pollution controls** Bio-heavy oil (BHO) is a renewable fuel, but its efficient use is problematic because its combustion may emit hazardous air pollutants (e.g., polycyclic aromatic hydrocarbon (PAH) compounds, NO x , and SO x ). Herein, catalytic fast pyrolysis over HZSM-5 zeolite was applied to upgrading BHO to drop-in fuel-range hydrocarbons with reduced contents of hazardous species such as PAH compounds and N- and S-containing species (NO x and SO x precursors). The effects of HZSM-5 desilication and linear low-density polyethylene (LLDPE) addition to the feedstock on hydrocarbon production were explored. The apparent activation energy for the thermal decomposition of BHO was up to 37.5 percent lowered by desilicated HZSM-5 (DeHZSM-5) compared with HZSM-5. Co-pyrolyzing LLDPE with BHO increased the content of drop-in fuel-range hydrocarbons and decreased the content of PAH compounds. The DeHZSM-5 was effective in producing drop-in fuel-range hydrocarbons from a mixture of BHO and LLDPE and suppressing the formation of N- and S-containing species and PAH compounds. The DeHZSM-5 enhanced the hydrocarbon production by up to 58.5 percent because of its enhanced porosity and high acid site density compared to its parent HZSM-5. This study experimentally validated that BHO can be upgraded to less hazardous fuel via catalytic fast co-pyrolysis with LLDPE over DeHZSM-5."""
    text = """**Suppression of the hazardous substances in catalytically upgraded bio-heavy oil as a precautious measure for clean air pollution controls** Bio-heavy oil (BHO) is a renewable fuel, but its efficient use is problematic because its combustion may emit hazardous air pollutants (e.g., polycyclic aromatic hydrocarbon (PAH) compounds, NO x , and SO x ). Herein, catalytic fast pyrolysis over HZSM-5 zeolite was applied to upgrading BHO to drop-in fuel-range hydrocarbons with reduced contents of hazardous species such as PAH compounds and N- and S-containing species (NO x and SO x precursors). The effects of HZSM-5 desilication and linear low-density polyethylene (LLDPE) addition to the feedstock on hydrocarbon production were explored. The apparent activation energy for the thermal decomposition of BHO was up to 37.5% lowered by desilicated HZSM-5 (DeHZSM-5) compared with HZSM-5. Co-pyrolyzing LLDPE with BHO increased the content of drop-in fuel-range hydrocarbons and decreased the content of PAH compounds. The DeHZSM-5 was effective in producing drop-in fuel-range hydrocarbons from a mixture of BHO and LLDPE and suppressing the formation of N- and S-containing species and PAH compounds. The DeHZSM-5 enhanced the hydrocarbon production by up to 58.5% because of its enhanced porosity and high acid site density compared to its parent HZSM-5. This study experimentally validated that BHO can be upgraded to less hazardous fuel via catalytic fast co-pyrolysis with LLDPE over DeHZSM-5."""
    # # Generate graph from wiki text with chunking

    graph = kg.generate(
        input_data=text,
        chunk_size=5000,
        context="renewable energies and biodiversity",
        cluster=True,
    )
    print("Entities:", graph.entities)
    print("Edges:", graph.edges)
    print("Relations:", graph.relations)

    # Log comprehensive usage summary
    kg.log_usage_summary()

    with open(
        f"tests/test_chunk_and_cluster-{Path(model).stem}.json", "w", encoding="utf-8"
    ) as f:
        f.write(graph.model_dump_json(indent=4))

    visualize(graph, f"tests/test_chunk_and_cluster-{Path(model).stem}.html")
