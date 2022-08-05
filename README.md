# AIBEDO 

AIBEDO is a hybrid AI framework to capture the effects of cloud properties on global circulation and regional climate patterns. 


## Documentation

For a detailed documentation of AIBEDO, on the model architecture, dashboard, datasets and scenarios, please visit https://aibedo.readthedocs.io/

## Environment

AIBEDO is developed in Python 3.9.

    # create new environment, if you want to use a different name, change the name (after -n) in the next line
    conda env create -f env_simple.yaml -n aibedo
    conda activate aibedo  # activate the environment called 'aibedo'

## Getting started

***Please have a look at [this README](aibedo/README.md)***

## Concept

Clouds play a vital role both in modulating Earth's radiation budget and shaping the coupled circulation of the atmosphere and ocean, driving regional changes in temperature and precipitation. The climate response to clouds is one of the largest uncertainties in state-of-the-art Earth System Models (ESMs) when producing decadal climate projections. This limitation becomes apparent when handling scenarios with large changes in cloud properties, e.g., 1) presence of greenhouse gases->loss of clouds or 2) engineered intervention like cloud brightening->increased cloud reflectivity.

Climate intervention techniques—like marine cloud brightening—that need to be fine-tuned spatiotemporally require thousands of hypothetical scenarios to find optimal strategies. Current ESMs need millions of core hours to complete a single simulation. AIBEDO is a hybrid AI model framework developed to resolve the weaknesses of ESMs by generating rapid and robust multi-decadal climate projections. We will demonstrate its utility using marine cloud brightening scenarios—to avoid climate tipping points and produce optimal intervention strategies


## Applications

- AIBEDO provides a robust framework to run multi-decadal climate scenarios that obviates the need to run ESM ensembles, saving millions of core computing hours. This tool therefore can be used for rapid generation of scenario analysis and for preparedness and climate resilience planning for military installations and operations.

- The multi-timescale nature of AIBEDO provides a rapid way to characterize tipping points at a decadal timescale and the early onset of tipping point (emerging trends) at weather timescale. 

- AIBEDO will develop optimal MCB climate intervention pathways to prevent dangerous tipping points in the climate system and can focus attention on those that destabilize regions and that have significant geopolitical impacts.

- AIBEDO also can be used to generate a portfolio of MCB-based intervention mitigation strategies that can be deployed in the event of 'surprise' regional climate interventions.

- The interactive nature of AIBEDO with parameter controls provides a unique tool for policymakers and modelers alike to perform ‘what-if’ investigations, run climate related scenarios, or identify consequences of planned, ill-planned, or rogue MCB experiments.


## Funding

The development of AIBEDO is funded under the DARPA AI-assisted Climate Tipping-point Modeling (ACTM) program under award DARPA-PA-21-04-02.

## Participating Institutions

- Palo Alto Research Center, Inc. (PARC)
- University of Victoria



