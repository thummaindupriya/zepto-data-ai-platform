\# Zepto Data \& AI Platform



A modular data and AI platform project containing a web-scraping data pipeline, Titanic analytics and machine-learning workflows, and a Zepto policy support assistant.



\## Repository Structure



```text

zepto-data-ai-platform/

│

├── data\_pipeline/

│   ├── scrape\_books.py

│   ├── database.py

│   ├── queries.py

│   ├── books.db

│   ├── books\_cleaned.csv

│   └── README.md

│

├── analytics/

│   ├── titanic\_analysis.py

│   ├── titanic.csv

│   ├── classifier\_comparison.csv

│   ├── imbalance\_comparison.csv

│   ├── random\_forest\_tuning.csv

│   ├── heteroscedasticity\_analysis.csv

│   ├── analysis\_recommendation.txt

│   ├── best\_classifier\_pipeline.joblib

│   ├── figures/

│   └── README.md

│

├── support\_assistant/

│   ├── docs/

│   │   ├── doc\_01.txt

│   │   ├── doc\_02.txt

│   │   ├── doc\_03.txt

│   │   ├── doc\_04.txt

│   │   ├── doc\_05.txt

│   │   ├── doc\_06.txt

│   │   ├── doc\_07.txt

│   │   └── doc\_08.txt

│   ├── ingest.py

│   ├── main.py

│   ├── prompts.py

│   ├── requirements.txt

│   ├── Dockerfile

│   └── README.md

│

└── README.md

```



\## Module 1 — Data Pipeline



Location: /data\_pipeline



This module demonstrates an end-to-end data pipeline using the Books to Scrape website.



\### Main tasks



\* Scrape book data using requests and BeautifulSoup

\* Collect title, price, star rating, availability, and category

\* Clean and transform the scraped data

\* Convert GBP prices to INR using the project exchange-rate assumption

\* Store the cleaned data in a normalized SQLite database

\* Execute SQL queries using filtering, sorting, limiting, distinct values, ranges, and joins

\* Verify SQL join results against an equivalent pandas merge



\### Main files



\* scrape\_books.py — web scraping and cleaning

\* database.py — SQLite database creation and loading

\* queries.py — SQL queries and pandas join verification

\* books\_cleaned.csv — cleaned dataset

\* books.db — SQLite database

\* README.md — module documentation



Run:



```powershell

python data\_pipeline/scrape\_books.py

python data\_pipeline/database.py

python data\_pipeline/queries.py

```



\---



\## Module 2 — Analytics



Location: `/analytics`



This module performs exploratory data analysis, visualization, classification, class-imbalance experiments, Random Forest tuning, and regression using the Titanic dataset.



\### Main workflow



1\. Load the Titanic dataset and save it as analytics/titanic.csv

2\. Generate a missing-value report

3\. Handle missing values using the assignment thresholds

4\. Perform univariate and bivariate analysis

5\. Analyze survival by sex, passenger class, and their combination

6\. Calculate correlations for the required six numeric variables

7\. Create multivariate visualizations with written interpretations

8\. Standardize age and fare using z-scores

9\. Build classification pipelines using:



&#x20;  \* Logistic Regression

&#x20;  \* Decision Tree

&#x20;  \* Random Forest

10\. Evaluate classifiers using:



\* Accuracy

\* Precision

\* Recall

\* F1

\* ROC-AUC

\* Confusion matrices



11\. Compare baseline Logistic Regression, balanced Logistic Regression, and SMOTE Logistic Regression

12\. Tune the Random Forest using GridSearchCV

13\. Calculate Random Forest out-of-bag (OOB) score

14\. Build a Linear Regression model for fare prediction

15\. Evaluate MAE, RMSE, R², and adjusted R²

16\. Analyze residuals and heteroscedasticity

17\. Save and reload the selected classifier pipeline



\### Main files



\* titanic\_analysis.py — complete analytics and modeling workflow

\* titanic.csv — saved Titanic dataset

\* classifier\_comparison.csv — classifier metrics

\* imbalance\_comparison.csv — class-imbalance experiment

\* random\_forest\_tuning.csv — Random Forest tuning results

\* heteroscedasticity\_analysis.csv — residual-spread analysis

\* analysis\_recommendation.txt — final written recommendation

\* best\_classifier\_pipeline.joblib — saved classifier pipeline

\* figures/ — generated charts

\* README.md — detailed analytics documentation



Run:



```powershell

python analytics/titanic\_analysis.py

```



\---



\## Module 3 — Support Assistant



Location: /support\_assistant



A local Zepto policy question-answering assistant using document embeddings, ChromaDB retrieval, LangGraph, Pydantic, and FastAPI.



\### Architecture



```text

8 Zepto policy documents

&#x20;       |

&#x20;       v

Document loading + chunking

&#x20;       |

&#x20;       v

Sentence Transformers

all-MiniLM-L6-v2

&#x20;       |

&#x20;       v

ChromaDB local vector store

&#x20;       |

&#x20;       v

FastAPI /ask

&#x20;       |

&#x20;       v

LangGraph StateGraph

&#x20;       |

&#x20;       +--> classify\_intent

&#x20;       |       |

&#x20;       |       +--> policy\_question

&#x20;       |       |       |

&#x20;       |       |       v

&#x20;       |       |   retrieve\_context

&#x20;       |       |       |

&#x20;       |       |       v

&#x20;       |       |   build\_prompt

&#x20;       |       |       |

&#x20;       |       |       v

&#x20;       |       |   retrieve\_and\_answer

&#x20;       |       |

&#x20;       |       +--> general\_question

&#x20;       |               |

&#x20;       |               v

&#x20;       |          direct\_answer

&#x20;       |

&#x20;       v

Pydantic validated response

```



\### Ingestion



The eight policy documents are stored under:



```text

support\_assistant/docs/

```



Run ingestion with:



```powershell

python support\_assistant/ingest.py

```



The ingestion process:



\* Loads all eight policy documents

\* Chunks the documents

\* Generates embeddings using all-MiniLM-L6-v2

\* Stores embeddings and metadata in a local ChromaDB collection named zepto\_policies



The current corpus produces 11 document chunks.



\### Prompt design



prompts.py contains a structured prompt with:



\* Role

\* Context

\* Task

\* Negative constraints

\* Output format

\* Length requirement

\* Few-shot example



The assistant is instructed to use only retrieved Zepto policy context and not invent policies, fees, timelines, or procedures.



\### LangGraph



main.py implements a StateGraph with named nodes for:



\* classify\_intent

\* retrieve\_context

\* build\_prompt

\* retrieve\_and\_answer

\* direct\_answer



The mock baseline uses deterministic keyword-based intent classification.



Policy questions are retrieved from ChromaDB using query embeddings and top-3 similarity retrieval.



General questions return:



```text

I can only answer questions about Zepto policies right now.

```



\### API



Start the FastAPI server:



```powershell

uvicorn support\_assistant.main:app --reload

```



The API endpoint is:



```text

POST /ask

```



Request:



```json

{

&#x20; "query": "What is the delivery fee for an order below INR 149?"

}

```



Example response:



```json

{

&#x20; "answer": "Based on the retrieved context: Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.",

&#x20; "sources": \[

&#x20;   "doc\_01",

&#x20;   "doc\_02",

&#x20;   "doc\_05"

&#x20; ],

&#x20; "confidence": 1.0

}

```



A general question such as:



```json

{

&#x20; "query": "What is the capital of India?"

}

```



returns the fixed general-question response with an empty source list.



\### Mock LLM mode



The assignment baseline uses deterministic mock behavior and does not require a network LLM.



The default grading mode is:



```text

MOCK\_LLM=1

```



The core implementation uses local embeddings and ChromaDB retrieval without requiring an external LLM service.



\### Docker



A Dockerfile is included at:



```text

support\_assistant/Dockerfile

```



The container is configured to run the FastAPI application locally.



\---



\## Technologies



\### Data Pipeline



\* Python

\* Requests

\* BeautifulSoup

\* pandas

\* SQLite



\### Analytics



\* Python

\* pandas

\* seaborn

\* matplotlib

\* scikit-learn

\* imbalanced-learn

\* joblib



\### Support Assistant



\* Python

\* Sentence Transformers

\* ChromaDB

\* LangGraph

\* Pydantic

\* FastAPI

\* Uvicorn

\* Docker



\## Reproducibility



Create and activate a Python virtual environment before running the modules.



Example:



```powershell

python -m venv .venv

.venv\\Scripts\\Activate.ps1

```



Install the required dependencies for the support assistant:



```powershell

pip install -r support\_assistant/requirements.txt

```



Each module contains its own README with module-specific implementation details and execution instructions.



\## Notes



Generated local files such as the ChromaDB database and Python cache files are excluded from Git using .gitignore.




## Submission Verification
The project includes all three required modules and a local FastAPI support assistant.

Docker note: the Dockerfile is included for local container execution.
