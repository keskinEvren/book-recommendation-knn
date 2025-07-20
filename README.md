# Book Recommendation System using K-Nearest Neighbors (KNN)

This is a book recommendation system that utilizes the K-Nearest Neighbors (KNN) algorithm to suggest books to users based on their past ratings and preferences.

## Key Features & Benefits

*   **Personalized Recommendations:** Provides book recommendations tailored to individual user tastes.
*   **KNN Algorithm:** Leverages the power of KNN to find users with similar reading habits.
*   **User-Friendly:** Simple and intuitive approach to book discovery.
*   **Data-Driven:** Based on real user ratings to ensure recommendation accuracy.

## Prerequisites & Dependencies

Before you begin, ensure you have the following installed:

*   **Python 3.x:** The core programming language.
*   **Jupyter Notebook:**  (Optional, but recommended) For running the provided notebook files.
*   **Required Python Libraries:** Install using `pip`:

    ```bash
    pip install pandas numpy scikit-learn
    ```

    Specifically you will need:

    *   `pandas`: For data manipulation and analysis.
    *   `numpy`: For numerical computations.
    *   `scikit-learn`: For the KNN algorithm and model evaluation.

## Installation & Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/keskinEvren/book-recommendation-knn.git
    cd book-recommendation-knn
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt  # if you create a requirements.txt file including the dependencies
    ```
    Or install them manually as per the Prerequisites section.

3.  **Download the Datasets:**

    The datasets (`Books.csv`, `Users.csv`, `Book-Ratings.csv`) should already be in the cloned repository. If not, you will need to obtain them.  (Note: Ensure proper attribution for data sources used.)

4.  **Run the Notebook (Optional):**

    Open `book_recommendation.ipynb` or `Untitled.ipynb` using Jupyter Notebook to explore and execute the recommendation system.

    ```bash
    jupyter notebook book_recommendation.ipynb
    ```

## Usage Examples & Code Overview

The core logic is implemented within the `book_recommendation.ipynb` notebook. Here's a brief overview:

1.  **Data Loading and Preprocessing:**
    *   Loads the book, user, and rating datasets using `pandas`.
    *   Merges the datasets into a unified structure.
    *   Handles missing values and data cleaning.

2.  **KNN Model Training:**
    *   Uses the `scikit-learn` library to implement the KNN algorithm.
    *   The model is trained on the user-item rating matrix.

3.  **Recommendation Generation:**
    *   Takes a user ID as input.
    *   Identifies the nearest neighbors (users with similar preferences).
    *   Recommends books based on the ratings of these neighbors.

Example Snippet (Illustrative):

```python
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Sample code - this is a simplified illustration.  Refer to the notebook for the full implementation
# Load the ratings data
ratings = pd.read_csv('Book-Ratings.csv')

# Create a user-item matrix
user_item_matrix = ratings.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating')

# Instantiate and fit the KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(user_item_matrix.fillna(0))

# Function to get recommendations (basic example)
def get_recommendations(user_id, model, data, n_recommendations=5):
    # Find the nearest neighbors to the specified user
    distances, indices = model.kneighbors([data.fillna(0).loc[user_id]], n_neighbors=n_recommendations+1)
    
    # ... (rest of the logic to extract book recommendations)
    return recommendations # a list of ISBNs
```

## Configuration Options

The main configuration options within the `book_recommendation.ipynb` notebook are:

*   **`n_neighbors`:**  The number of neighbors to consider in the KNN algorithm. You can adjust this parameter to fine-tune the recommendation accuracy and diversity.
*   **`metric`:**  The distance metric used by the KNN algorithm (e.g., 'cosine', 'euclidean'). 'Cosine' similarity is generally preferred for recommender systems.
*   **Data File Paths:** Ensure that the paths to the data files (`Books.csv`, `Users.csv`, `Book-Ratings.csv`) are correctly specified in the notebook.

## Contributing Guidelines

Contributions are welcome! Here's how you can contribute:

1.  **Fork the Repository:** Create your own fork of the repository.
2.  **Create a Branch:** Create a new branch for your feature or bug fix.
3.  **Make Changes:** Implement your changes and commit them with clear, descriptive messages.
4.  **Submit a Pull Request:** Submit a pull request to the main branch of the original repository.

Please ensure your code adheres to the project's coding style and includes relevant tests.

## License Information

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Acknowledgments

*   The datasets used in this project are from [Insert Original Source of Datasets Here, if applicable].
*   Thanks to the open-source community for providing the `pandas`, `numpy`, and `scikit-learn` libraries.
