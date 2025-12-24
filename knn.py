# 4️⃣ KNN – Movie Recommendation Based on Similar Users
# Real-time problem

# “Show movies liked by users similar to you.”

# Simplified version:

# Each user has features like:

# average rating of action movies

# average rating of romance

# average rating of comedy

# You join similar users and recommend what they liked.
# Why KNN?

# KNN directly uses similarity in feature space.

# For simple recommendation prototypes, you can:

# Represent users by preference scores

# Recommend based on nearest neighbors.

# Note: Real recommender systems use more advanced methods (matrix factorization, deep models), but KNN is good to explain.
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Each row is a user: [action_score, romance_score, comedy_score]
X = np.array([
    [5, 1, 2],  # likes action
    [4, 1, 1],
    [1, 5, 4],  # likes romance+comedy
    [1, 4, 5],
    [3, 3, 3],  # mixed
])

# label = 1 -> recommend "Action Movie A", 0 -> don't recommend
y = np.array([1, 1, 0, 0, 1])

# 1️⃣ Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# 2️⃣ Create and train KNN on TRAIN data only
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 3️⃣ Evaluate on TEST data
y_pred = knn.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))

# 4️⃣ Use the model for a NEW user
new_user = np.array([[4, 2, 2]])  # likes action a bit, romance a bit
print(
    "Recommend Action Movie A? (1=yes,0=no):",
    knn.predict(new_user)[0]
)
