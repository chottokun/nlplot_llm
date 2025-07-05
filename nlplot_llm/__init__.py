from .core import NLPlotLLM

# Bringing other potential utilities or constants if they were in nlplot.nlplot directly
# For now, assuming NLPlotLLM is the main export.
# If other functions from the original nlplot.nlplot were meant to be top-level,
# they would need to be imported from .core as well. E.g.:
# from .core import (
#     NLPlotLLM,
#     _ranked_topics_for_edges, # If this was intended for package-level access
#     get_colorpalette,
#     generate_freq_df
# )
# However, typically helper functions like _ranked_topics_for_edges are not exported.
# get_colorpalette and generate_freq_df might be, depending on design.
# For now, only exporting the main class.

__version__ = "0.1.0" # Initialize version for the new package

def main():
    """
    Placeholder main function, can be developed or removed.
    """
    print(f"nlplot_llm version {__version__}")

# To make functions like generate_freq_df directly available from nlplot_llm import:
# from .core import generate_freq_df
# from .core import get_colorpalette
# This makes them accessible as nlplot_llm.generate_freq_df()
# If they are not added here, they would be nlplot_llm.core.generate_freq_df()
# Let's make them available at package level for ease of use if they are general utilities.
from .core import get_colorpalette, generate_freq_df
