==========================
Frequently Asked Questions
==========================

.. warning::

    - This project is still under active development. 
    - The codebase is being rewritten from a notebook-based internal use prototype to a more modular and extensible package. 
    - As such, the API and functionalities may change significantly in future releases. 
    - Users are advised to check the documentation and release notes for updates.

This section addresses some common questions about the LoGoMOD package.

1. Architecture & Build 
==========================

The slides mention Python (Pandas, CuDF), DuckDB, GPU optional. Are there any additional libraries, services, or system dependencies not shown in the presentation?

    As the package depends on other libraries for a lot of its work, there will naturally be other dependencies. 

Is there a diagram or workflow map of how the modules interact (data flow from input → processing → output/front-end)? If not, can one be provided? 

    Yes, this will be added into the workflow documentation. 

2. Repository & Version Control 
==========================

We saw GitHub forking, branches, and pull requests outlined. Are there agreed coding standards or contribution guidelines beyond what’s in the slides? 

    Standard Git rules apply. Since ALGA has no plans to continue maintaining the codebase, upstream contributions and downstream pulls on forks might be necessary.

Are there automated tests or CI/CD in place? 

    Not currently, but in the works.  The codebase is still being restructured. Once done, the aim is to have pipelines to test and publish PyPI/Conda packages.

3. Data Inputs 
==========================

ABS Census Microdata Sample Files are listed. What other datasets are pre-configured or required? 

    Other datasets at the core include TableBuilder margins and Center for Population projections. Given that custom behaviour needs to be implemented by users, the package is to be seen as a framework rather than a turnkey solution.
 
What preprocessing scripts exist for raw data, and where are they located in the repo? 

    All preprocessing needed is done within the package itself. There are no separate scripts. 

How often should core data sources be refreshed? 

    This depends on the data source. For example, Census data is updated every 5 years, while population projections may be updated more frequently.

4. Model Functionality 
==========================

Which modules are already complete and validated, and which are experimental or incomplete? 

    There are no complete modules in the package at the moment. The codebase is being restructured from a prototype to a more modular package. Users are welcome to write their own modules and contribute back.

What assumptions and calibration steps were applied at LGA level? 

    Regression analysis was used to generate significant relationships between demographic variables at the LGA level. Calibration was done using known margins from TableBuilder.

    An extension of Cathal O'Donoghue's Quota Sampling algorithm from his work on spatial microsimulation is used to sample individuals into the synthetic population.

Are there adjustable parameters or configuration files for scenario testing (population growth, housing, disaster shocks)? 

    Yes, users can define their own parameters and configurations for different scenarios. The package is designed to be flexible and extensible to accommodate various use cases.

5. Frontend & Integration 
==========================

Is there an existing user-facing frontend connected, or is everything run from Python notebooks/scripts? 
    
    There is no frontend currently. The package is intended to be used as a library within Python scripts or notebooks. But, the output data can be easily integrated into dashboards or other visualization tools.

If outputs are meant for dashboards, which formats are currently generated (CSV, DuckDB queries, APIs)? 

    The synthetic population dataset will be output as a DuckDB store. However, different output writers will be implemented soon.

Has LoGoMOD been integrated with visualisation tools (e.g., Power BI, Tableau)? 

    Not yet, but since the output is in DuckDB format, it can be easily connected to various visualization tools that support DuckDB as a data source.

6. Operations & Maintenance 
==========================

What is the standard process to execute a model run end-to-end? 

    The standard process involves setting up the model with the required data inputs, defining the parameters for the simulation, and then running the model to generate the synthetic population. Further analyses and modelling can be performed on the output data as needed.

Approximate runtime with CPU vs GPU? 

    This will depend on the size of the dataset and the complexity of the simulation. Generally, using a GPU can significantly speed up computations, especially for large datasets. However, specific benchmarks are not yet available as the package is still under development.

Known failure points or bottlenecks? 

    As the package is still being developed, there may be performance bottlenecks that have not yet been identified. Users are encouraged to report any issues they encounter.

Are there monitoring/logging scripts? 

    Not currently, but logging functionality will be added in future releases to help users monitor model runs and debug issues.

7. Governance & Collaboration 
==========================

Beyond “fork, branch, contribute back” outlined in the slides, are there rules for module ownership or review? 

    There are no formal rules for module ownership or review at the moment. However, users are encouraged to follow best practices for code contributions, including code reviews and testing.

Are there existing derivative models run by councils we should be aware of? 

    None

8. Knowledge Transfer 
==========================

Is there full documentation or a runbook beyond the PowerPoint and GitHub README? 

    Yes, documentation is being developed and will be made available on the project's GitHub repository.

Are there worked examples of simulations (e.g., case studies or tutorials)? 

    Not yet, but tutorials and case studies will be added to the documentation.

Any undocumented scripts, credentials, or datasets tied to the developer’s account? 

    None

9. Future Development & Risks 
==========================

What enhancements were planned but not started? 

    There are plans to add more modules for different demographic dynamics, improve performance, and enhance the user experience. Specific enhancements will be determined based on user feedback and contributions.

What gaps/limitations should we be aware of? 

    Currently, the package may not fully support all demographic factors or complex interactions between variables. Users should be aware of these limitations when designing their simulations.

What skills are essential to maintain (e.g., Python, GPU, microsimulation design, frontend dev)? 

    Key skills include proficiency in Python, understanding of microsimulation design principles, and familiarity with data manipulation and analysis libraries. Knowledge of GPU programming may also be beneficial for optimizing performance.
