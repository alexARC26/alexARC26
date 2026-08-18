# Comprehensive Data Science & ML Portfolio
**Contact:** [🌐 alex-collado.com](https://alex-collado.com) | [:envelope: alejandrorodriguezcollado@gmail.com](mailto:alejandrorodriguezcollado@gmail.com) | [:link: LinkedIn](https://www.linkedin.com/in/alejandro-rodr%C3%ADguez-collado-a3456b17a)

I’m Alejandro Rodríguez-Collado, a senior Data Scientist and Data & AI Tech Lead with 8+ years of experience delivering data-driven products and AI solutions across real estate, biomedicine, and Industry 4.0.

Proven track record leading cross-functional teams of up to 7 engineers and scientists, building scalable data platforms, and translating business needs into production-grade analytics, machine learning, and LLM-powered solutions. Ph.D. in Statistics combining strong technical depth with project leadership, stakeholder management, and strategic execution.

## About me
- **Data Science & AI**: Machine Learning (ML), Deep Learning (DL), Computer Vision, Statistical Modeling, Time Series Analysis, Data Visualization.
- **ML/DL Frameworks**: Scikit-learn, TensorFlow, PyTorch.
- **Generative AI & LLMs**: Retrieval-Augmented Generation (RAG), Agentic RAG (LangGraph), LangChain (LCEL), Vector Databases (Chroma), Embeddings (Sentence-Transformers, Hugging Face), LLM Inference (OpenAI, local via Ollama), RAG Evaluation, Prompt Engineering.
- **Data Engineering**: ETL / ELT, Data Pipelines, Snowflake, Apache Spark, Apache Hop, Apache Airflow.
- **Programming Languages**: Python, SQL, R, Java.
- **Cloud**: Amazon Web Services (AWS), Google Cloud Platform (GCP), Microsoft Azure, Databricks.
- **Visualization Tools**: Power BI, Grafana, Tableau.
- **Leadership & Delivery**: Team Management, Agile Project Management, Stakeholder Management.
- **Languages**: Spanish (Native), English (Fluent / Near native; certified by Cambridge University).

## Project Overview
| Project | Tool | Objective | Key Result | Link |
|---------|-----------|-----------|------------|------|
| **CO2 Emissions** | Python | Predict vehicle emissions with regression models | Tuned XGBoost with $R^2$ of 0.998 | [portfolio-regression-data-viz](https://github.com/alexARC26/portfolio-regression-data-viz/tree/main) |
| **Cats-vs-Dogs** | Python | Classification of cat and dog images | Inception CNN with accuracy of 0.967 | [portfolio-classification-neural-networks](https://github.com/alexARC26/portfolio-classification-neural-networks/tree/main) |
| **RAG Basics** | Python | Build and evaluate a Retrieval-Augmented Generation pipeline | Hit Rate@10 of 88.89% (MRR 0.633), verified faithfulness via LLM-as-judge | [rag-basics](https://github.com/alexARC26/rag-basics) |

## Projects
### 1. CO₂ Emissions: Analysis and Regression Prediction (Python)

Conducted a comprehensive analysis to predict CO2 emissions from a dataset of 7,385 Canadian vehicles. The project consists of two notebooks: (1) Exploratory data analysis, preprocessing, and visualization to clean the dataset, engineer features, and visualize relationships using boxplots, histograms, scatterplots, and correlation heatmaps; (2) Regression modeling to predict emissions using linear regression, LASSO, support vector machines, regression trees, and XGBoost. XGBoost, after hyperparameter tuning, achieved the best performance (0.998 $R^2$ and 7.400 MSE). This model could inform eco-friendly vehicle policies or consumer decision-making.
- **Techniques**: 
    - Preprocessing: Cleaning, feature engineering, normalization, categorical encoding.
    - Analysis: Univariate, bivariate and multivariate analysis, correlation analysis, visualization.
    - Modeling: Linear regression, LASSO regularization, SVM, regression trees, XGBoost.
    - Evaluation: Train-validation-test split, feature importance analysis, hyperparameter tuning.
- **Tools**: Python. Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (LinearRegression, SVR, DecisionTreeRegressor), Statsmodels, XGBoost, Kagglehub.
- **Link**: [portfolio-regression-data-viz](https://github.com/alexARC26/portfolio-regression-data-viz/tree/main)

![Model Performance by R2 and MSE](https://raw.githubusercontent.com/alexARC26/portfolio-regression-data-viz/main/images/Results_Summary.png)
*Figure 1: Model performance for CO₂ emissions prediction, evaluated by R² and MSE.*

### 2. Cats-vs.-Dogs Image Classification (Python)

Developed a model to classify cat and dog images using a dataset with a high feature-to-sample ratio (2,000 images and 67,500 features per image). The project includes preprocessing and three notebooks, each exploring a different approach: (1) Classic ML with dimensionality reduction techniques; (2) Sequential CNNs; (3) Inception CNNs with transfer learning. The latter achieved 0.967 accuracy and 0.970 F1 score in an indepedent test dataset, outperforming other models. This approach could support wildlife monitoring or pet identification systems.
- **Techniques**: 
    - Preprocessing: Image processing, feature engineering, normalization, dimensionality reduction, principal component analysis.
    - Modeling: Logistic regression, decision trees, random forest, sequential CNN, Inception CNN, transfer learning.
    - Evaluation: Train-validation-test split, hyperparameter tuning.
- **Tools**: Python. Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (logistic regression, decision tree, and random forest), Scitkit-image, Keras, Tensorflow, PCA.
- **Link**: [portfolio-classification-neural-networks](https://github.com/alexARC26/portfolio-classification-neural-networks/tree/main)

![Model Performance by accuracy and F1 score](https://raw.githubusercontent.com/alexARC26/portfolio-classification-neural-networks/main/images/Results_Summary.png)
*Figure 2: Model performance for cats-vs.-dogs classification, evaluated by accuracy and F1 score.*

### 3. RAG Basics: Retrieval-Augmented Generation Pipeline (Python)

Built a progressive series of notebooks covering the full RAG lifecycle: from a minimal retriever to an agentic system with quantitative evaluation. Starting with document chunking and a Chroma vector store, the project builds a Retrieve → Augment → Generate chain (first with a local Ollama `llama3.2` model, then with real PDF ingestion and LCEL), converts it into an agentic RAG system where the LLM decides when to call retrieval as a tool (LangGraph), and finally evaluates the system quantitatively — retrieval quality (Hit Rate, MRR) and generation quality (LLM-as-judge faithfulness scoring). At k=10, the retriever achieved an 88.89% Hit Rate and 0.633 MRR; the LLM judge confirmed high faithfulness on answered questions and correctly refused adversarial, out-of-context questions instead of hallucinating.
- **Techniques**:
    - Indexing: document chunking, embeddings, vector storage.
    - Generation: RAG chains with LCEL, prompt engineering, agentic retrieval (tool-calling).
    - Evaluation: retrieval metrics (Hit Rate, MRR), LLM-as-judge faithfulness scoring, adversarial testing.
- **Tools**: Python. LangChain, LangGraph, Chroma, Sentence-Transformers / Hugging Face embeddings, Ollama (local `llama3.2`), OpenAI API, PyPDF.
- **Link**: [rag-basics](https://github.com/alexARC26/rag-basics)

![Hit Rate and MRR by k](https://raw.githubusercontent.com/alexARC26/rag-basics/main/images/Results_Summary.png)
*Figure 3: RAG retrieval quality by k, evaluated by Hit Rate and MRR.*

### 3. Upcoming Projects 
Currently developing projects to further expand the portfolio — Stay tuned for updates!

## Previous Work
Highlights of published contributions from past roles:
- **CRAN R Package (2020–2022)**: Tech lead and maintainer of a package for the functional data analysis of oscillatory signals. [Link](https://cran.r-project.org/web/packages/FMM).
- **Shiny Apps with R (2019–2022)**: Led development of interactive web apps for data visualization and analysis in biomedicine and neuroscience. Example: [Cre Line’s Neuronal APs](https://alexarc26.shinyapps.io/median_ap_profile_by_cre_line).

## How to Explore
- Each project has a dedicated `README.md` with detailed descriptions, code, and results.
- Notebooks are located in the `notebooks/` folder of each project and are optimized for seamless execution in Google Colab, with integrated data downloads and dependencies for immediate reproducibility.
- Explore the repository on GitHub for the latest updates.

Thank you for visiting my portfolio! For suggestions or collaboration opportunities, please contact with me on [🌐 alex-collado.com](https://alex-collado.com).
